#include <gtest/gtest.h>

#include <libbase/point.h>
#include <libbase/stats.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/shared_device_image.h>

#include "test_utils.h"
#include "kernels/kernels.h"
#include "kernels/defines.h"

namespace {

void preparePipelineCaches(gpu::Context &context, size_t capacity_per_family)
{
	context.vk()->setPipelineCacheCapacityPerFamily(capacity_per_family);
	context.vk()->clearPipelineCaches();
	context.vk()->resetComputePipelineCacheStats();
	context.vk()->resetRasterPipelineCacheStats();
}

void expectBufferValue(const gpu::gpu_mem_32u &buffer, size_t index, unsigned int expected_value)
{
	std::vector<unsigned int> values = buffer.readVector();
	ASSERT_LT(index, values.size());
	EXPECT_EQ(values[index], expected_value);
}

void launchWriteValueKernel(gpu::gpu_mem_32u &buffer0, gpu::gpu_mem_32u &buffer1, unsigned int chosen_buffer, int index, unsigned int value)
{
	avk2::KernelSource kernel(avk2::getWriteValueAtIndexKernel());
	struct {
		unsigned int chosenBuffer;
		int index;
		unsigned int value;
	} params = {chosen_buffer, index, value};
	kernel.exec(params, gpu::WorkSize(VK_GROUP_SIZE, 1), buffer0, buffer1);
}

void launchCopyKernel(gpu::gpu_mem_32u &buffer_dst, gpu::gpu_mem_32u &buffer_src, unsigned int n)
{
	avk2::KernelSource kernel(avk2::getFillBufferWithZerosKernel());
	kernel.exec(n, gpu::WorkSize(VK_GROUP_SIZE, n), buffer_dst, buffer_src);
}

void createRectangleGeometry(gpu::gpu_mem_vertices_xyz &gpu_vertices, gpu::gpu_mem_32u &gpu_faces)
{
	std::vector<point3f> vertices_xyz(4);
	vertices_xyz[0] = point3f(0.15f, 0.25f, 0.20f);
	vertices_xyz[1] = point3f(0.75f, 0.25f, 0.60f);
	vertices_xyz[2] = point3f(0.15f, 0.65f, 0.20f);
	vertices_xyz[3] = point3f(0.75f, 0.65f, 0.60f);

	std::vector<point3u> faces(2);
	faces[0] = point3u(0, 1, 2);
	faces[1] = point3u(2, 1, 3);

	gpu_vertices.resizeN(vertices_xyz.size());
	gpu_vertices.writeN((gpu::Vertex3D*) vertices_xyz.data(), vertices_xyz.size());

	gpu_faces.resizeN(3 * faces.size());
	gpu_faces.writeN((unsigned int*) faces.data(), 3 * faces.size());
}

void checkRectangleRendering(const image8u &color_image, const image32i &face_id_image, const image32f &depth_image, float empty_depth)
{
	ASSERT_EQ(color_image.channels(), 3);
	ASSERT_EQ(face_id_image.width(), color_image.width());
	ASSERT_EQ(depth_image.height(), color_image.height());

	ptrdiff_t inside_i = color_image.width() / 2;
	ptrdiff_t inside_j = color_image.height() / 2;
	for (ptrdiff_t c = 0; c < color_image.channels(); ++c) {
		EXPECT_EQ(color_image.ptr(inside_j)[color_image.channels() * inside_i + c], SOME_COLOR_VALUE + c);
	}
	EXPECT_IN_RANGE(face_id_image.ptr(inside_j)[inside_i], 0, 1);
	EXPECT_IN_RANGE(depth_image.ptr(inside_j)[inside_i], 0.2f, 0.6f);

	ptrdiff_t outside_i = 0;
	ptrdiff_t outside_j = 0;
	for (ptrdiff_t c = 0; c < color_image.channels(); ++c) {
		EXPECT_EQ(color_image.ptr(outside_j)[color_image.channels() * outside_i + c], 0);
	}
	EXPECT_EQ(face_id_image.ptr(outside_j)[outside_i], std::numeric_limits<int>::max());
	EXPECT_EQ(depth_image.ptr(outside_j)[outside_i], empty_depth);
}

void launchRasterRectangle(size_t width, size_t height)
{
	gpu::gpu_mem_vertices_xyz gpu_vertices;
	gpu::gpu_mem_32u gpu_faces;
	createRectangleGeometry(gpu_vertices, gpu_faces);

	const float empty_depth = 1.0f;
	const int no_face_index = std::numeric_limits<int>::max();

	gpu::shared_device_depth_image depth_buffer(width, height);
	gpu_image8u color_buffer(width, height, 3);
	gpu_image32i face_id_buffer(width, height);

	avk2::KernelSource kernel(avk2::getRasterizeKernel());
	point4uc zero_clear_color = {0, 0, 0, 0};
	kernel.initRender(width, height).geometry(gpu_vertices, gpu_faces)
		.setDepthAttachment(depth_buffer, empty_depth)
		.addAttachment(color_buffer, zero_clear_color)
		.addAttachment(face_id_buffer, no_face_index)
		.params(2)
		.exec();

	checkRectangleRendering(color_buffer.read(), face_id_buffer.read(), depth_buffer.read(), empty_depth);
}

void launchRasterRectangleWithBlending(size_t width, size_t height)
{
	gpu::gpu_mem_vertices_xyz gpu_vertices;
	gpu::gpu_mem_32u gpu_faces;
	createRectangleGeometry(gpu_vertices, gpu_faces);

	gpu_image32f color_buffer(width, height, 4);
	color_buffer.fill(0.0f);

	avk2::KernelSource kernel(avk2::getRasterizeWithBlendingKernel());
	kernel.initRender(width, height).geometry(gpu_vertices, gpu_faces)
		.addAttachment(color_buffer)
		.setColorAttachmentsBlending(true)
		.exec();

	image32f image = color_buffer.read();
	ptrdiff_t inside_i = image.width() / 2;
	ptrdiff_t inside_j = image.height() / 2;
	EXPECT_EQ(image.ptr(inside_j)[4 * inside_i + 0], BLENDING_RED_VALUE);
	EXPECT_EQ(image.ptr(inside_j)[4 * inside_i + 1], BLENDING_GREEN_VALUE);
	EXPECT_EQ(image.ptr(inside_j)[4 * inside_i + 2], BLENDING_BLUE_VALUE);
	EXPECT_EQ(image.ptr(inside_j)[4 * inside_i + 3], BLENDING_ALPHA_VALUE);
	ptrdiff_t outside_i = 0;
	ptrdiff_t outside_j = 0;
	for (int c = 0; c < 4; ++c) {
		EXPECT_EQ(image.ptr(outside_j)[4 * outside_i + c], 0.0f);
	}
}

}

TEST(vulkan, computePipelineCacheTracksHitsAndMisses)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		preparePipelineCaches(context, 1);

		const unsigned int n = 8;
		gpu::gpu_mem_32u buffer0(n), buffer1(n), buffer_src(n), buffer_dst(n);
		buffer0.fill(0);
		buffer1.fill(0);
		std::vector<unsigned int> src_values(n, 7);
		buffer_src.writeN(src_values.data(), n);
		buffer_dst.fill(0);

		launchWriteValueKernel(buffer0, buffer1, 0, 1, 11);
		expectBufferValue(buffer0, 1, 11);

		launchWriteValueKernel(buffer0, buffer1, 1, 2, 22);
		expectBufferValue(buffer1, 2, 22);

		launchCopyKernel(buffer_dst, buffer_src, n);
		expectBufferValue(buffer_dst, 3, 7);

		launchCopyKernel(buffer_dst, buffer_src, n);
		expectBufferValue(buffer_dst, 5, 7);

		avk2::PipelineCacheStats stats = context.vk()->getComputePipelineCacheStats();
		EXPECT_EQ(stats.misses, 2);
		EXPECT_EQ(stats.hits, 2);
		EXPECT_EQ(context.vk()->computePipelineCacheSize(), 2);
		EXPECT_EQ(context.vk()->cachedPipelinesCount(), 2);
		context.vk()->logComputePipelineCacheContents();
	}

	checkPostInvariants();
}

TEST(vulkan, computeKernelWorksWithDifferentArguments)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.vk()->clearComputePipelineCache();

		const unsigned int n = 8;
		gpu::gpu_mem_32u buffer0(n), buffer1(n);
		buffer0.fill(0);
		buffer1.fill(0);

		launchWriteValueKernel(buffer0, buffer1, 0, 1, 11);
		expectBufferValue(buffer0, 1, 11);
		expectBufferValue(buffer1, 1, 0);

		launchWriteValueKernel(buffer0, buffer1, 1, 2, 22);
		expectBufferValue(buffer0, 1, 11);
		expectBufferValue(buffer1, 2, 22);

		launchWriteValueKernel(buffer0, buffer1, 0, 7, 77);
		expectBufferValue(buffer0, 7, 77);
		expectBufferValue(buffer1, 2, 22);
	}

	checkPostInvariants();
}

TEST(vulkan, rasterPipelineCacheTracksHitsAndMisses)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		preparePipelineCaches(context, 2);

		launchRasterRectangle(64, 48);
		launchRasterRectangle(96, 72);
		launchRasterRectangleWithBlending(64, 48);
		launchRasterRectangleWithBlending(96, 72);

		avk2::PipelineCacheStats stats = context.vk()->getRasterPipelineCacheStats();
		EXPECT_EQ(stats.misses, 2);
		EXPECT_EQ(stats.hits, 2);
		EXPECT_EQ(context.vk()->rasterPipelineCacheSize(), 2);
		EXPECT_EQ(context.vk()->cachedPipelinesCount(), 2);
		context.vk()->logRasterPipelineCacheContents();
	}

	checkPostInvariants();
}

TEST(vulkan, computePipelineCacheStressShowsSignificantSpeedup)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		preparePipelineCaches(context, 4);

		gpu::gpu_mem_32u buffer0(1), buffer1(1);
		buffer0.fill(0);
		buffer1.fill(0);

		const int niters = 12;
		std::vector<double> uncached_preparing_times;
		std::vector<double> cached_preparing_times;
		uncached_preparing_times.reserve(niters);
		cached_preparing_times.reserve(niters);

		for (int iter = 0; iter < niters; ++iter) {
			context.vk()->clearComputePipelineCache();
			avk2::KernelSource kernel(avk2::getWriteValueAtIndexKernel());
			struct {
				unsigned int chosenBuffer;
				int index;
				unsigned int value;
			} params = {0u, 0, narrow_cast<unsigned int>(iter + 1)};
			kernel.exec(params, gpu::WorkSize(VK_GROUP_SIZE, 1), buffer0, buffer1);
			uncached_preparing_times.push_back(kernel.getLastExecPrepairingTime());
		}

		context.vk()->clearComputePipelineCache();
		context.vk()->resetComputePipelineCacheStats();
		for (int iter = 0; iter < niters; ++iter) {
			avk2::KernelSource kernel(avk2::getWriteValueAtIndexKernel());
			struct {
				unsigned int chosenBuffer;
				int index;
				unsigned int value;
			} params = {1u, 0, narrow_cast<unsigned int>(100 + iter)};
			kernel.exec(params, gpu::WorkSize(VK_GROUP_SIZE, 1), buffer0, buffer1);
			cached_preparing_times.push_back(kernel.getLastExecPrepairingTime());
		}

		std::vector<double> cached_hits_only(cached_preparing_times.begin() + 1, cached_preparing_times.end());
		double uncached_median = stats::median(uncached_preparing_times);
		double cached_median = stats::median(cached_hits_only);
		double speedup_ratio = uncached_median / cached_median;

		std::cout << "uncached preparing time: " << stats::valuesStatsLine(uncached_preparing_times) << std::endl;
		std::cout << "cached preparing time: " << stats::valuesStatsLine(cached_preparing_times) << std::endl;
		std::cout << "cached-hit preparing time: " << stats::valuesStatsLine(cached_hits_only) << std::endl;
		std::cout << "compute cache stress speedup ratio=" << speedup_ratio << std::endl;

		avk2::PipelineCacheStats cache_stats = context.vk()->getComputePipelineCacheStats();
		EXPECT_EQ(cache_stats.misses, 1);
		EXPECT_EQ(cache_stats.hits, niters - 1);
		EXPECT_EQ(context.vk()->computePipelineCacheSize(), 1);
		EXPECT_GE(speedup_ratio, 2.0);
	}

	checkPostInvariants();
}
