#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <libbase/point.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/shared_device_image.h>
#include <libgpu/vulkan/data_buffer.h>
#include <libgpu/vulkan/vulkan_api_headers.h>

#include "../../../../../src/io/scene_reader.h"

#include "test_utils.h"
#include "kernels/kernels.h"

namespace {

constexpr uint32_t kGroupSizeX = 16;
constexpr uint32_t kGroupSizeY = 16;
constexpr vk::Format kPackedHdrFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::Format kBc6hHdrFormat = vk::Format::eBc6HUfloatBlock;

struct ReductionMoments {
	double sum = 0.0;
	double sum_sq = 0.0;
	uint64_t count = 0;
};

struct CompressionMetrics {
	std::vector<double> total_seconds;
	std::vector<double> pack_seconds;
	std::vector<double> upload_seconds;
};

struct ImageDiffStats {
	double rms_brightness_error = 0.0;
	uint64_t valid_pixels = 0;
};

struct BrightnessStatsResult {
	double mean = 0.0;
	double stddev = 0.0;
	uint64_t valid_pixels = 0;
	std::vector<double> kernel_seconds;
};

class PackedHdrPackedImage : public gpu::shared_device_image {
protected:
	avk2::raii::ImageData* allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type) override
	{
		rassert(cn == 1, 2026032915355600001, cn);
		rassert(data_type == DataType64u, 2026032915355600002, typeName(data_type));
		gpu::Context context;
		rassert(context.type() == gpu::Context::TypeVulkan, 2026032915355600003);
		const vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eTransferDst
											  | vk::ImageUsageFlagBits::eTransferSrc
											  | vk::ImageUsageFlagBits::eSampled;
		return context.vk()->createImage2DArray(width, height, 1, kPackedHdrFormat, usage_flags);
	}
};

class PackedHdrBc6hImage : public gpu::shared_device_image {
protected:
	avk2::raii::ImageData* allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type) override
	{
		rassert(cn == 1, 2026032915203900001, cn);
		rassert(data_type == DataType8u, 2026032915203900002, typeName(data_type));
		gpu::Context context;
		rassert(context.type() == gpu::Context::TypeVulkan, 2026032915203900003);
		const vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eTransferDst
											  | vk::ImageUsageFlagBits::eTransferSrc
											  | vk::ImageUsageFlagBits::eSampled;
		return context.vk()->createImage2DArray(width, height, 1, kBc6hHdrFormat, usage_flags);
	}
};

std::filesystem::path findRepoRoot()
{
	namespace fs = std::filesystem;
	fs::path current = fs::current_path();
	for (;;) {
		if (fs::exists(current / ".gitignore") && fs::exists(current / "libs") && fs::exists(current / "data")) {
			return current;
		}
		if (current == current.root_path()) {
			break;
		}
		current = current.parent_path();
	}
	rassert(false, 2026032915355600004, fs::current_path().string());
	return {};
}

bool isLlvmPipeDevice(const gpu::Device &device)
{
	return device.name.find("llvmpipe") != std::string::npos || device.name.find("lavapipe") != std::string::npos;
}

bool supportsPackedHdrFormat(gpu::Context &context)
{
	const vk::FormatProperties format_properties = context.vk()->getPhysicalDevice().getFormatProperties(kPackedHdrFormat);
	const vk::FormatFeatureFlags required = vk::FormatFeatureFlagBits::eTransferDst
										  | vk::FormatFeatureFlagBits::eTransferSrc
										  | vk::FormatFeatureFlagBits::eSampledImage;
	return (format_properties.optimalTilingFeatures & required) == required;
}

bool supportsBc6hFormat(gpu::Context &context)
{
	const vk::FormatProperties format_properties = context.vk()->getPhysicalDevice().getFormatProperties(kBc6hHdrFormat);
	const vk::FormatFeatureFlags required = vk::FormatFeatureFlagBits::eTransferDst
										  | vk::FormatFeatureFlagBits::eTransferSrc
										  | vk::FormatFeatureFlagBits::eSampledImage;
	return (format_properties.optimalTilingFeatures & required) == required;
}

std::vector<point3f> normalizeSceneToViewport(const SceneGeometry &scene)
{
	rassert(!scene.vertices.empty(), 2026032915355600005);
	point3f min_pt = scene.vertices.front();
	point3f max_pt = scene.vertices.front();
	for (const point3f &v : scene.vertices) {
		for (int c = 0; c < 3; ++c) {
			min_pt[c] = std::min(min_pt[c], v[c]);
			max_pt[c] = std::max(max_pt[c], v[c]);
		}
	}

	const point3f center = (min_pt + max_pt) * 0.5f;
	const float range_x = std::max(max_pt.x - min_pt.x, 1.0e-6f);
	const float range_y = std::max(max_pt.y - min_pt.y, 1.0e-6f);
	const float range_z = std::max(max_pt.z - min_pt.z, 1.0e-6f);
	const float scale_xy = 0.88f / std::max(range_x, range_y);

	std::vector<point3f> out(scene.vertices.size());
	for (size_t i = 0; i < scene.vertices.size(); ++i) {
		const point3f &src = scene.vertices[i];
		point3f dst;
		dst.x = 0.5f + (src.x - center.x) * scale_xy;
		dst.y = 0.5f - (src.y - center.y) * scale_xy;
		dst.z = 0.10f + 0.80f * (src.z - min_pt.z) / range_z;
		out[i] = dst;
	}
	return out;
}

gpu::gpu_mem_vertices_xyz uploadVertices(const std::vector<point3f> &vertices_xyz)
{
	std::vector<gpu::Vertex3D> vertices_gpu(vertices_xyz.size());
	for (size_t i = 0; i < vertices_xyz.size(); ++i) {
		vertices_gpu[i].init(vertices_xyz[i].x, vertices_xyz[i].y, vertices_xyz[i].z);
	}
	gpu::gpu_mem_vertices_xyz gpu_vertices(vertices_xyz.size());
	gpu_vertices.writeN(vertices_gpu.data(), vertices_gpu.size());
	return gpu_vertices;
}

gpu::gpu_mem_faces_indices uploadFaces(const std::vector<point3u> &faces)
{
	gpu::gpu_mem_faces_indices gpu_faces(faces.size());
	gpu_faces.writeN(faces.data(), faces.size());
	return gpu_faces;
}

ReductionMoments reduceMoments(const std::vector<float> &sum,
							   const std::vector<float> &sum_sq,
							   const std::vector<uint32_t> &count)
{
	rassert(sum.size() == sum_sq.size() && sum.size() == count.size(), 2026032915355600006, sum.size(), sum_sq.size(), count.size());
	ReductionMoments result;
	for (size_t i = 0; i < sum.size(); ++i) {
		result.sum += sum[i];
		result.sum_sq += sum_sq[i];
		result.count += count[i];
	}
	return result;
}

double computeStdDev(const ReductionMoments &moments)
{
	rassert(moments.count > 0, 2026032915355600007);
	const double inv_count = 1.0 / static_cast<double>(moments.count);
	const double mean = moments.sum * inv_count;
	const double variance = std::max(0.0, moments.sum_sq * inv_count - mean * mean);
	return std::sqrt(variance);
}

double computeMegapixelsPerSecond(size_t width, size_t height, double seconds)
{
	rassert(seconds > 0.0, 2026032915355600008, seconds);
	return static_cast<double>(width) * static_cast<double>(height) / 1.0e6 / seconds;
}

double bytesToMiB(size_t bytes)
{
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

template <typename Fn>
std::vector<double> runTimedIterations(size_t iterations, Fn &&fn)
{
	std::vector<double> seconds;
	seconds.reserve(iterations);
	for (size_t i = 0; i < iterations; ++i) {
		timer t;
		fn();
		seconds.push_back(t.elapsed());
	}
	return seconds;
}

BrightnessStatsResult runBrightnessStatsFloat(avk2::KernelSource &kernel,
											  gpu::Context &context,
											  gpu_image32f &original_image,
											  size_t width,
											  size_t height,
											  size_t iterations)
{
	const size_t groups_x = div_ceil(width, static_cast<size_t>(kGroupSizeX));
	const size_t groups_y = div_ceil(height, static_cast<size_t>(kGroupSizeY));
	const size_t partial_count = groups_x * groups_y;
	gpu::gpu_mem_32f partial_sum(partial_count);
	gpu::gpu_mem_32f partial_sum_sq(partial_count);
	gpu::gpu_mem_32u partial_count_gpu(partial_count);
	const gpu::WorkSize work_size(kGroupSizeX, kGroupSizeY, width, height);
	struct PushConstants {
		uint32_t width;
		uint32_t height;
		uint32_t groups_x;
	} params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(groups_x)};

	kernel.exec(params, work_size, original_image, partial_sum, partial_sum_sq, partial_count_gpu);
	context.vk()->waitForAllInflightComputeLaunches();
	partial_sum.fill(0.0f);
	partial_sum_sq.fill(0.0f);
	partial_count_gpu.fill(0u);

	BrightnessStatsResult result;
	result.kernel_seconds = runTimedIterations(iterations, [&]() {
		partial_sum.fill(0.0f);
		partial_sum_sq.fill(0.0f);
		partial_count_gpu.fill(0u);
		kernel.exec(params, work_size, original_image, partial_sum, partial_sum_sq, partial_count_gpu);
		context.vk()->waitForAllInflightComputeLaunches();
	});

	const ReductionMoments moments = reduceMoments(partial_sum.readVector(), partial_sum_sq.readVector(), partial_count_gpu.readVector());
	result.valid_pixels = moments.count;
	result.mean = moments.sum / static_cast<double>(moments.count);
	result.stddev = computeStdDev(moments);
	return result;
}

BrightnessStatsResult runBrightnessStatsPacked(avk2::KernelSource &kernel,
											   gpu::Context &context,
											   PackedHdrPackedImage &packed_image,
											   size_t width,
											   size_t height,
											   size_t iterations)
{
	const size_t groups_x = div_ceil(width, static_cast<size_t>(kGroupSizeX));
	const size_t groups_y = div_ceil(height, static_cast<size_t>(kGroupSizeY));
	const size_t partial_count = groups_x * groups_y;
	gpu::gpu_mem_32f partial_sum(partial_count);
	gpu::gpu_mem_32f partial_sum_sq(partial_count);
	gpu::gpu_mem_32u partial_count_gpu(partial_count);
	const gpu::WorkSize work_size(kGroupSizeX, kGroupSizeY, width, height);
	struct PushConstants {
		uint32_t width;
		uint32_t height;
		uint32_t groups_x;
	} params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(groups_x)};

	kernel.exec(params, work_size, packed_image, partial_sum, partial_sum_sq, partial_count_gpu);
	context.vk()->waitForAllInflightComputeLaunches();
	partial_sum.fill(0.0f);
	partial_sum_sq.fill(0.0f);
	partial_count_gpu.fill(0u);

	BrightnessStatsResult result;
	result.kernel_seconds = runTimedIterations(iterations, [&]() {
		partial_sum.fill(0.0f);
		partial_sum_sq.fill(0.0f);
		partial_count_gpu.fill(0u);
		kernel.exec(params, work_size, packed_image, partial_sum, partial_sum_sq, partial_count_gpu);
		context.vk()->waitForAllInflightComputeLaunches();
	});

	const ReductionMoments moments = reduceMoments(partial_sum.readVector(), partial_sum_sq.readVector(), partial_count_gpu.readVector());
	result.valid_pixels = moments.count;
	result.mean = moments.sum / static_cast<double>(moments.count);
	result.stddev = computeStdDev(moments);
	return result;
}

BrightnessStatsResult runBrightnessStatsPacked(avk2::KernelSource &kernel,
											   gpu::Context &context,
											   PackedHdrBc6hImage &packed_image,
											   size_t width,
											   size_t height,
											   size_t iterations)
{
	const size_t groups_x = div_ceil(width, static_cast<size_t>(kGroupSizeX));
	const size_t groups_y = div_ceil(height, static_cast<size_t>(kGroupSizeY));
	const size_t partial_count = groups_x * groups_y;
	gpu::gpu_mem_32f partial_sum(partial_count);
	gpu::gpu_mem_32f partial_sum_sq(partial_count);
	gpu::gpu_mem_32u partial_count_gpu(partial_count);
	const gpu::WorkSize work_size(kGroupSizeX, kGroupSizeY, width, height);
	struct PushConstants {
		uint32_t width;
		uint32_t height;
		uint32_t groups_x;
	} params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(groups_x)};

	kernel.exec(params, work_size, packed_image, partial_sum, partial_sum_sq, partial_count_gpu);
	context.vk()->waitForAllInflightComputeLaunches();
	partial_sum.fill(0.0f);
	partial_sum_sq.fill(0.0f);
	partial_count_gpu.fill(0u);

	BrightnessStatsResult result;
	result.kernel_seconds = runTimedIterations(iterations, [&]() {
		partial_sum.fill(0.0f);
		partial_sum_sq.fill(0.0f);
		partial_count_gpu.fill(0u);
		kernel.exec(params, work_size, packed_image, partial_sum, partial_sum_sq, partial_count_gpu);
		context.vk()->waitForAllInflightComputeLaunches();
	});

	const ReductionMoments moments = reduceMoments(partial_sum.readVector(), partial_sum_sq.readVector(), partial_count_gpu.readVector());
	result.valid_pixels = moments.count;
	result.mean = moments.sum / static_cast<double>(moments.count);
	result.stddev = computeStdDev(moments);
	return result;
}

ImageDiffStats runImageDiffKernel(avk2::KernelSource &kernel,
								  gpu::Context &context,
								  gpu_image32f &original_image,
								  PackedHdrPackedImage &packed_image,
								  size_t width,
								  size_t height)
{
	const size_t groups_x = div_ceil(width, static_cast<size_t>(kGroupSizeX));
	const size_t groups_y = div_ceil(height, static_cast<size_t>(kGroupSizeY));
	const size_t partial_count = groups_x * groups_y;
	gpu::gpu_mem_32f partial_sse(partial_count);
	gpu::gpu_mem_32u partial_count_gpu(partial_count);
	const gpu::WorkSize work_size(kGroupSizeX, kGroupSizeY, width, height);
	struct PushConstants {
		uint32_t width;
		uint32_t height;
		uint32_t groups_x;
	} params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(groups_x)};

	partial_sse.fill(0.0f);
	partial_count_gpu.fill(0u);
	kernel.exec(params, work_size, original_image, packed_image, partial_sse, partial_count_gpu);
	context.vk()->waitForAllInflightComputeLaunches();

	const std::vector<float> sse = partial_sse.readVector();
	const std::vector<uint32_t> count = partial_count_gpu.readVector();
	double total_sse = 0.0;
	uint64_t total_count = 0;
	for (size_t i = 0; i < sse.size(); ++i) {
		total_sse += sse[i];
		total_count += count[i];
	}
	rassert(total_count > 0, 2026032915355600009);
	return {std::sqrt(total_sse / static_cast<double>(total_count)), total_count};
}

ImageDiffStats runImageDiffKernel(avk2::KernelSource &kernel,
								  gpu::Context &context,
								  gpu_image32f &original_image,
								  PackedHdrBc6hImage &packed_image,
								  size_t width,
								  size_t height)
{
	const size_t groups_x = div_ceil(width, static_cast<size_t>(kGroupSizeX));
	const size_t groups_y = div_ceil(height, static_cast<size_t>(kGroupSizeY));
	const size_t partial_count = groups_x * groups_y;
	gpu::gpu_mem_32f partial_sse(partial_count);
	gpu::gpu_mem_32u partial_count_gpu(partial_count);
	const gpu::WorkSize work_size(kGroupSizeX, kGroupSizeY, width, height);
	struct PushConstants {
		uint32_t width;
		uint32_t height;
		uint32_t groups_x;
	} params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(groups_x)};

	partial_sse.fill(0.0f);
	partial_count_gpu.fill(0u);
	kernel.exec(params, work_size, original_image, packed_image, partial_sse, partial_count_gpu);
	context.vk()->waitForAllInflightComputeLaunches();

	const std::vector<float> sse = partial_sse.readVector();
	const std::vector<uint32_t> count = partial_count_gpu.readVector();
	double total_sse = 0.0;
	uint64_t total_count = 0;
	for (size_t i = 0; i < sse.size(); ++i) {
		total_sse += sse[i];
		total_count += count[i];
	}
	rassert(total_count > 0, 2026032915203900004);
	return {std::sqrt(total_sse / static_cast<double>(total_count)), total_count};
}

} // namespace

TEST(vulkan, hdrPackedImageBenchmark)
{
	namespace fs = std::filesystem;

	const fs::path repo_root = findRepoRoot();
	const fs::path gnome_path = repo_root / "data" / "gnome" / "gnome.ply";
	ASSERT_TRUE(fs::exists(gnome_path)) << "Missing gnome scene at " << gnome_path;

	const SceneGeometry scene = loadPLY(gnome_path.string());
	const std::vector<point3f> normalized_vertices = normalizeSceneToViewport(scene);

	size_t tested_devices = 0;
	for (gpu::Device device : enumVKDevices()) {
		std::cout << "Testing Vulkan device: " << device.name << std::endl;
		gpu::Context context = activateVKContext(device);
		if (!supportsPackedHdrFormat(context)) {
			std::cout << "Skipping device because " << vk::to_string(kPackedHdrFormat)
					  << " is not supported for sampled transfer images" << std::endl;
			continue;
		}
		++tested_devices;

		const bool is_llvmpipe = isLlvmPipeDevice(device);
		const size_t width = is_llvmpipe ? 512 : 2048;
		const size_t height = is_llvmpipe ? 512 : 2048;
		const size_t iterations = is_llvmpipe ? 2 : 5;

		gpu::gpu_mem_vertices_xyz gpu_vertices = uploadVertices(normalized_vertices);
		gpu::gpu_mem_faces_indices gpu_faces = uploadFaces(scene.faces);
		gpu::shared_device_depth_image depth_buffer(width, height);
		gpu_image32f original_image(width, height, 3);
		PackedHdrPackedImage packed_image;
		packed_image.resize(width, height, 1, DataType64u);
		gpu::gpu_mem_32u packed_buffer(width * height * 2);

		avk2::KernelSource raster_kernel(avk2::getRasterizeHdrGnomeKernel());
		avk2::KernelSource pack_kernel(avk2::getPackHdrRgb9e5Kernel());
		avk2::KernelSource diff_kernel(avk2::getCompareHdrRgb9e5Kernel());
		avk2::KernelSource brightness_float_kernel(avk2::getBrightnessStatsFloatKernel());
		avk2::KernelSource brightness_packed_kernel(avk2::getBrightnessStatsRgb9e5Kernel());

		const point4f clear_color(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
		raster_kernel.initRender(width, height).geometry(gpu_vertices, gpu_faces)
			.setDepthAttachment(depth_buffer, 1.0f)
			.addAttachment(original_image, clear_color)
			.exec();
		context.vk()->waitForAllInflightRenderLaunches();

		struct PackPushConstants {
			uint32_t width;
			uint32_t height;
		} pack_params{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
		const gpu::WorkSize pack_work_size(kGroupSizeX, kGroupSizeY, width, height);

		pack_kernel.exec(pack_params, pack_work_size, original_image, packed_buffer);
		context.vk()->waitForAllInflightComputeLaunches();
		packed_image.write(packed_buffer, width, height, 0, 0, 0, false);

		CompressionMetrics compression_metrics;
		compression_metrics.total_seconds.reserve(iterations);
		compression_metrics.pack_seconds.reserve(iterations);
		compression_metrics.upload_seconds.reserve(iterations);
		for (size_t iter = 0; iter < iterations; ++iter) {
			packed_buffer.fill(0u);
			timer total_timer;
			timer pack_timer;
			pack_kernel.exec(pack_params, pack_work_size, original_image, packed_buffer);
			context.vk()->waitForAllInflightComputeLaunches();
			compression_metrics.pack_seconds.push_back(pack_timer.elapsed());

			timer upload_timer;
			packed_image.write(packed_buffer, width, height, 0, 0, 0, false);
			context.vk()->waitForAllInflightComputeLaunches();
			compression_metrics.upload_seconds.push_back(upload_timer.elapsed());
			compression_metrics.total_seconds.push_back(total_timer.elapsed());
		}

		const ImageDiffStats diff_stats = runImageDiffKernel(diff_kernel, context, original_image, packed_image, width, height);
		const BrightnessStatsResult brightness_original = runBrightnessStatsFloat(brightness_float_kernel, context, original_image, width, height, iterations);
		const BrightnessStatsResult brightness_packed = runBrightnessStatsPacked(brightness_packed_kernel, context, packed_image, width, height, iterations);

		EXPECT_GT(diff_stats.valid_pixels, width * height / 8);
		EXPECT_LT(diff_stats.valid_pixels, width * height);
		EXPECT_TRUE(std::isfinite(diff_stats.rms_brightness_error));
		EXPECT_LT(diff_stats.rms_brightness_error, 0.05);
		EXPECT_TRUE(std::isfinite(brightness_original.stddev));
		EXPECT_TRUE(std::isfinite(brightness_packed.stddev));
		EXPECT_NEAR(brightness_original.stddev, brightness_packed.stddev, 0.05);
		EXPECT_EQ(brightness_original.valid_pixels, width * height);
		EXPECT_EQ(brightness_packed.valid_pixels, width * height);

		const double original_size_mib = bytesToMiB(width * height * 3 * sizeof(float));
		const double packed_size_mib = bytesToMiB(packed_image.size());
		const double compression_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(compression_metrics.total_seconds));
		const double pack_only_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(compression_metrics.pack_seconds));
		const double brightness_original_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(brightness_original.kernel_seconds));
		const double brightness_packed_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(brightness_packed.kernel_seconds));

		std::cout << "HDR packed benchmark: " << width << "x" << height
				  << " valid_pixels=" << diff_stats.valid_pixels
				  << " original_float32_rgb=" << original_size_mib << " MiB"
				  << " packed_rgba16f=" << packed_size_mib << " MiB"
				  << " compression_ratio=" << (original_size_mib / packed_size_mib)
				  << std::endl;
		std::cout << "  compression total median:  " << stats::median(compression_metrics.total_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(compression_metrics.total_seconds) << ")"
				  << " throughput=" << compression_mpix_s << " MPix/s" << std::endl;
		std::cout << "  pack kernel median:        " << stats::median(compression_metrics.pack_seconds) * 1000.0 << " ms"
				  << " throughput=" << pack_only_mpix_s << " MPix/s" << std::endl;
		std::cout << "  buffer->image median:      " << stats::median(compression_metrics.upload_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(compression_metrics.upload_seconds) << ")" << std::endl;
		std::cout << "  rms brightness error:      " << diff_stats.rms_brightness_error << std::endl;
		std::cout << "  brightness stddev original:" << " sigma=" << brightness_original.stddev
				  << " mean=" << brightness_original.mean
				  << " median=" << stats::median(brightness_original.kernel_seconds) * 1000.0 << " ms"
				  << " throughput=" << brightness_original_mpix_s << " MPix/s" << std::endl;
		std::cout << "  brightness stddev packed:  " << " sigma=" << brightness_packed.stddev
				  << " mean=" << brightness_packed.mean
				  << " median=" << stats::median(brightness_packed.kernel_seconds) * 1000.0 << " ms"
				  << " throughput=" << brightness_packed_mpix_s << " MPix/s" << std::endl;
	}

	EXPECT_GT(tested_devices, 0u);
	checkPostInvariants();
}

TEST(vulkan, bc6hImageBenchmark)
{
	namespace fs = std::filesystem;

	const fs::path repo_root = findRepoRoot();
	const fs::path gnome_path = repo_root / "data" / "gnome" / "gnome.ply";
	ASSERT_TRUE(fs::exists(gnome_path)) << "Missing gnome scene at " << gnome_path;

	const SceneGeometry scene = loadPLY(gnome_path.string());
	const std::vector<point3f> normalized_vertices = normalizeSceneToViewport(scene);

	size_t tested_devices = 0;
	for (gpu::Device device : enumVKDevices()) {
		std::cout << "Testing Vulkan device: " << device.name << std::endl;
		gpu::Context context = activateVKContext(device);
		if (!supportsBc6hFormat(context)) {
			std::cout << "Skipping device because " << vk::to_string(kBc6hHdrFormat)
					  << " is not supported for sampled transfer images" << std::endl;
			continue;
		}
		++tested_devices;

		const bool is_llvmpipe = isLlvmPipeDevice(device);
		const size_t width = is_llvmpipe ? 512 : 2048;
		const size_t height = is_llvmpipe ? 512 : 2048;
		const size_t iterations = is_llvmpipe ? 2 : 5;
		rassert(width % 4 == 0 && height % 4 == 0, 2026032915305200001, width, height);

		gpu::gpu_mem_vertices_xyz gpu_vertices = uploadVertices(normalized_vertices);
		gpu::gpu_mem_faces_indices gpu_faces = uploadFaces(scene.faces);
		gpu::shared_device_depth_image depth_buffer(width, height);
		gpu_image32f original_image(width, height, 3);
		PackedHdrBc6hImage bc6h_image;
		bc6h_image.resize(width, height, 1, DataType8u);
		const size_t blocks_x = width / 4;
		const size_t blocks_y = height / 4;
		gpu::gpu_mem_32u bc6h_blocks(blocks_x * blocks_y * 4);

		avk2::KernelSource raster_kernel(avk2::getRasterizeHdrGnomeKernel());
		avk2::KernelSource encode_kernel(avk2::getEncodeBc6hConstantKernel());
		avk2::KernelSource diff_kernel(avk2::getCompareHdrRgb9e5Kernel());
		avk2::KernelSource brightness_float_kernel(avk2::getBrightnessStatsFloatKernel());
		avk2::KernelSource brightness_packed_kernel(avk2::getBrightnessStatsRgb9e5Kernel());

		const point4f clear_color(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
		raster_kernel.initRender(width, height).geometry(gpu_vertices, gpu_faces)
			.setDepthAttachment(depth_buffer, 1.0f)
			.addAttachment(original_image, clear_color)
			.exec();
		context.vk()->waitForAllInflightRenderLaunches();

		struct EncodePushConstants {
			uint32_t width;
			uint32_t height;
			uint32_t blocks_x;
		} encode_params{static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(blocks_x)};
		const gpu::WorkSize encode_work_size(8, 8, blocks_x, blocks_y);

		encode_kernel.exec(encode_params, encode_work_size, original_image, bc6h_blocks);
		context.vk()->waitForAllInflightComputeLaunches();
		bc6h_image.write(bc6h_blocks, width, height, 0, 0, 0, false);

		CompressionMetrics compression_metrics;
		compression_metrics.total_seconds.reserve(iterations);
		compression_metrics.pack_seconds.reserve(iterations);
		compression_metrics.upload_seconds.reserve(iterations);
		for (size_t iter = 0; iter < iterations; ++iter) {
			bc6h_blocks.fill(0u);
			timer total_timer;
			timer pack_timer;
			encode_kernel.exec(encode_params, encode_work_size, original_image, bc6h_blocks);
			context.vk()->waitForAllInflightComputeLaunches();
			compression_metrics.pack_seconds.push_back(pack_timer.elapsed());

			timer upload_timer;
			bc6h_image.write(bc6h_blocks, width, height, 0, 0, 0, false);
			context.vk()->waitForAllInflightComputeLaunches();
			compression_metrics.upload_seconds.push_back(upload_timer.elapsed());
			compression_metrics.total_seconds.push_back(total_timer.elapsed());
		}

		const ImageDiffStats diff_stats = runImageDiffKernel(diff_kernel, context, original_image, bc6h_image, width, height);
		const BrightnessStatsResult brightness_original = runBrightnessStatsFloat(brightness_float_kernel, context, original_image, width, height, iterations);
		const BrightnessStatsResult brightness_bc6h = runBrightnessStatsPacked(brightness_packed_kernel, context, bc6h_image, width, height, iterations);

		EXPECT_GT(diff_stats.valid_pixels, width * height / 8);
		EXPECT_LT(diff_stats.valid_pixels, width * height);
		EXPECT_TRUE(std::isfinite(diff_stats.rms_brightness_error));
		EXPECT_LT(diff_stats.rms_brightness_error, 0.05);
		EXPECT_TRUE(std::isfinite(brightness_original.stddev));
		EXPECT_TRUE(std::isfinite(brightness_bc6h.stddev));
		EXPECT_NEAR(brightness_original.stddev, brightness_bc6h.stddev, 0.05);
		EXPECT_EQ(brightness_original.valid_pixels, width * height);
		EXPECT_EQ(brightness_bc6h.valid_pixels, width * height);

		const double original_size_mib = bytesToMiB(width * height * 3 * sizeof(float));
		const double bc6h_size_mib = bytesToMiB(blocks_x * blocks_y * 16);
		const double compression_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(compression_metrics.total_seconds));
		const double pack_only_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(compression_metrics.pack_seconds));
		const double brightness_original_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(brightness_original.kernel_seconds));
		const double brightness_bc6h_mpix_s = computeMegapixelsPerSecond(width, height, stats::median(brightness_bc6h.kernel_seconds));

		std::cout << "BC6H benchmark: " << width << "x" << height
				  << " valid_pixels=" << diff_stats.valid_pixels
				  << " original_float32_rgb=" << original_size_mib << " MiB"
				  << " bc6h=" << bc6h_size_mib << " MiB"
				  << " compression_ratio=" << (original_size_mib / bc6h_size_mib)
				  << std::endl;
		std::cout << "  compression total median:  " << stats::median(compression_metrics.total_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(compression_metrics.total_seconds) << ")"
				  << " throughput=" << compression_mpix_s << " MPix/s" << std::endl;
		std::cout << "  bc6h encode median:        " << stats::median(compression_metrics.pack_seconds) * 1000.0 << " ms"
				  << " throughput=" << pack_only_mpix_s << " MPix/s" << std::endl;
		std::cout << "  buffer->image median:      " << stats::median(compression_metrics.upload_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(compression_metrics.upload_seconds) << ")" << std::endl;
		std::cout << "  rms brightness error:      " << diff_stats.rms_brightness_error << std::endl;
		std::cout << "  brightness stddev original:" << " sigma=" << brightness_original.stddev
				  << " mean=" << brightness_original.mean
				  << " median=" << stats::median(brightness_original.kernel_seconds) * 1000.0 << " ms"
				  << " throughput=" << brightness_original_mpix_s << " MPix/s" << std::endl;
		std::cout << "  brightness stddev bc6h:    " << " sigma=" << brightness_bc6h.stddev
				  << " mean=" << brightness_bc6h.mean
				  << " median=" << stats::median(brightness_bc6h.kernel_seconds) * 1000.0 << " ms"
				  << " throughput=" << brightness_bc6h_mpix_s << " MPix/s" << std::endl;
	}

	EXPECT_GT(tested_devices, 0u);
	checkPostInvariants();
}
