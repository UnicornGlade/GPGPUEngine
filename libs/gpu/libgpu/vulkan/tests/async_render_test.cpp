#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/stats.h>
#include <libbase/point.h>
#include <libgpu/shared_device_buffer.h>

#include <cstdlib>
#include <optional>

#include "test_utils.h"

#include "kernels/defines.h"
#include "kernels/kernels.h"

namespace {

struct ScopedEnvVar {
	std::string name;
	std::optional<std::string> old_value;

	ScopedEnvVar(const std::string &name_, const std::string &value) : name(name_)
	{
		const char *existing = std::getenv(name.c_str());
		if (existing != nullptr) {
			old_value = std::string(existing);
		}
#ifdef _WIN32
		_putenv_s(name.c_str(), value.c_str());
#else
		setenv(name.c_str(), value.c_str(), 1);
#endif
	}

	~ScopedEnvVar()
	{
		if (old_value.has_value()) {
#ifdef _WIN32
			_putenv_s(name.c_str(), old_value->c_str());
#else
			setenv(name.c_str(), old_value->c_str(), 1);
#endif
		} else {
#ifdef _WIN32
			_putenv_s(name.c_str(), "");
#else
			unsetenv(name.c_str());
#endif
		}
	}
};

struct RenderBenchmarkResult {
	double elapsed_seconds = 0.0;
	avk2::AsyncRenderStats stats;
	image32f image;
};

std::pair<gpu::gpu_mem_vertices_xyz, gpu::gpu_mem_32u> createRectangleGeometry(float from_x, float to_x, float from_y, float to_y)
{
	std::vector<point3f> vertices_xyz(4);
	float rect_z = 0.50f;
	vertices_xyz[0] = point3f(from_x, from_y, rect_z);
	vertices_xyz[1] = point3f(to_x,   from_y, rect_z);
	vertices_xyz[2] = point3f(from_x, to_y,   rect_z);
	vertices_xyz[3] = point3f(to_x,   to_y,   rect_z);

	std::vector<point3u> faces(2);
	faces[0] = point3u(0, 1, 2);
	faces[1] = point3u(2, 1, 3);

	gpu::gpu_mem_vertices_xyz gpu_vertices;
	gpu_vertices.resizeN(vertices_xyz.size());
	gpu_vertices.writeN((gpu::Vertex3D *) vertices_xyz.data(), vertices_xyz.size());

	gpu::gpu_mem_32u gpu_faces;
	gpu_faces.resizeN(3 * faces.size());
	gpu_faces.writeN((unsigned int *) faces.data(), 3 * faces.size());

	return {gpu_vertices, gpu_faces};
}

bool areTimestampQueriesEnabled()
{
	const char *value = std::getenv("GPGPU_ENABLE_VULKAN_TIMESTAMP_QUERIES");
	if (value == nullptr) {
		return true;
	}
	std::string normalized = tolower(trimmed(value));
	return !(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no");
}

bool isAsyncRenderEnabled()
{
	const char *value = std::getenv("GPGPU_ENABLE_VULKAN_ASYNC_RENDER");
	if (value == nullptr) {
		return true;
	}
	std::string normalized = tolower(trimmed(value));
	return !(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no");
}

bool isNvidiaDevice(const gpu::Device &device)
{
	return tolower(device.name).find("nvidia") != std::string::npos;
}

RenderBenchmarkResult runBlendingBenchmark(const gpu::Device &device, bool async_enabled, size_t width, size_t height, int nlaunches, size_t max_inflight)
{
	ScopedEnvVar async_render_env("GPGPU_ENABLE_VULKAN_ASYNC_RENDER", async_enabled ? "true" : "false");

	gpu::Context context = activateVKContext(const_cast<gpu::Device &>(device), true);
	context.setMemoryGuardsChecksAfterKernels(false);
	context.vk()->setMaxInflightRenderLaunches(max_inflight);

	gpu_image32f colors_accumulator_buffer(width, height, 4);
	colors_accumulator_buffer.fill(0.0f);

	auto [gpu_vertices, gpu_faces] = createRectangleGeometry(0.45f, 0.55f, 0.45f, 0.55f);
	avk2::KernelSource kernel_rasterization_with_blending(avk2::getRasterizeWithBlendingKernel());

	context.vk()->resetAsyncRenderStats();
	timer timer_chain;
	timer_chain.start();
	for (int launch = 0; launch < nlaunches; ++launch) {
		kernel_rasterization_with_blending.initRender(width, height).geometry(gpu_vertices, gpu_faces)
			.addAttachment(colors_accumulator_buffer).setColorAttachmentsBlending(true)
			.exec();
	}
	RenderBenchmarkResult result;
	result.elapsed_seconds = timer_chain.elapsed();
	result.stats = context.vk()->getAsyncRenderStats(true);
	result.image = colors_accumulator_buffer.read();
	return result;
}

void expectBlendingCenterAndCorner(const image32f &image, int nlaunches)
{
	rassert(image.channels() == 4, 2026031517415800016);
	size_t center_x = image.width() / 2;
	size_t center_y = image.height() / 2;
	size_t corner_x = image.width() / 8;
	size_t corner_y = image.height() / 8;

	const float expected[4] = {
		BLENDING_RED_VALUE * nlaunches,
		BLENDING_GREEN_VALUE * nlaunches,
		BLENDING_BLUE_VALUE * nlaunches,
		BLENDING_ALPHA_VALUE * nlaunches,
	};
	for (int c = 0; c < 4; ++c) {
		EXPECT_FLOAT_EQ(image.ptr(center_y)[4 * center_x + c], expected[c]);
		EXPECT_FLOAT_EQ(image.ptr(corner_y)[4 * corner_x + c], 0.0f);
	}
}

} // namespace

TEST(vulkan, asyncRenderBlendingChainAccumulatesCorrectly)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.setMemoryGuardsChecksAfterKernels(false);
		context.vk()->setMaxInflightRenderLaunches(32);
		context.vk()->resetAsyncRenderStats();

		const size_t width = 512;
		const size_t height = 512;
		const int nlaunches = 64;
		gpu_image32f colors_accumulator_buffer(width, height, 4);
		colors_accumulator_buffer.fill(0.0f);

		auto [gpu_vertices, gpu_faces] = createRectangleGeometry(0.25f, 0.75f, 0.25f, 0.75f);
		avk2::KernelSource kernel_rasterization_with_blending(avk2::getRasterizeWithBlendingKernel());

		for (int launch = 0; launch < nlaunches; ++launch) {
			kernel_rasterization_with_blending.initRender(width, height).geometry(gpu_vertices, gpu_faces)
				.addAttachment(colors_accumulator_buffer).setColorAttachmentsBlending(true)
				.exec();
		}

		avk2::AsyncRenderStats stats = context.vk()->getAsyncRenderStats(true);
		context.vk()->logAsyncRenderStats("asyncRenderBlendingChainAccumulatesCorrectly", false);
		if (isAsyncRenderEnabled() && areTimestampQueriesEnabled()) {
			EXPECT_GT(stats.total_seconds, 0.0);
			EXPECT_EQ(stats.launches_count, static_cast<size_t>(nlaunches));
		} else if (!isAsyncRenderEnabled()) {
			EXPECT_EQ(stats.launches_count, 0u);
		}

		image32f image = colors_accumulator_buffer.read();
		expectBlendingCenterAndCorner(image, nlaunches);
	}

	checkPostInvariants();
}

TEST(vulkan, asyncRenderBlendingChainRespectsInflightLimit)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.setMemoryGuardsChecksAfterKernels(false);
		context.vk()->setMaxInflightRenderLaunches(2);
		context.vk()->resetAsyncRenderStats();

		const size_t width = 256;
		const size_t height = 256;
		const int nlaunches = 128;
		gpu_image32f colors_accumulator_buffer(width, height, 4);
		colors_accumulator_buffer.fill(0.0f);

		auto [gpu_vertices, gpu_faces] = createRectangleGeometry(0.40f, 0.60f, 0.40f, 0.60f);
		avk2::KernelSource kernel_rasterization_with_blending(avk2::getRasterizeWithBlendingKernel());

		for (int launch = 0; launch < nlaunches; ++launch) {
			kernel_rasterization_with_blending.initRender(width, height).geometry(gpu_vertices, gpu_faces)
				.addAttachment(colors_accumulator_buffer).setColorAttachmentsBlending(true)
				.exec();
		}

		avk2::AsyncRenderStats stats = context.vk()->getAsyncRenderStats(true);
		context.vk()->logAsyncRenderStats("asyncRenderBlendingChainRespectsInflightLimit", false);
		EXPECT_GT(stats.launches_count, 0u);
		EXPECT_GT(stats.max_inflight_observed, 0u);
		EXPECT_LE(stats.max_inflight_observed, 2u);

		image32f image = colors_accumulator_buffer.read();
		expectBlendingCenterAndCorner(image, nlaunches);
		context.vk()->setMaxInflightRenderLaunches(32);
	}

	checkPostInvariants();
}

TEST(vulkan, asyncRenderBlendingChainShowsSpeedup)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		const size_t width = 512;
		const size_t height = 512;
		const int nlaunches = 512;

		RenderBenchmarkResult sync = runBlendingBenchmark(devices[k], false, width, height, nlaunches, 32);
		RenderBenchmarkResult async = runBlendingBenchmark(devices[k], true, width, height, nlaunches, 32);

		double speedup = sync.elapsed_seconds / async.elapsed_seconds;
		std::cout << "asyncRenderBlendingChainShowsSpeedup: sync=" << sync.elapsed_seconds
				  << " s, async=" << async.elapsed_seconds
				  << " s, speedup=" << speedup << "x" << std::endl;
		expectBlendingCenterAndCorner(sync.image, nlaunches);
		expectBlendingCenterAndCorner(async.image, nlaunches);

		if (isNvidiaDevice(devices[k])) {
			EXPECT_GT(speedup, 1.2);
		}
	}

	checkPostInvariants();
}
