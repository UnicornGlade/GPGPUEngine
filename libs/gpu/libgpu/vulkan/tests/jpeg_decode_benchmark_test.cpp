#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <jpeglib.h>

#include <libbase/nvtx_markers.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "test_utils.h"

namespace {

namespace fs = std::filesystem;

constexpr uint32_t kBenchmarkIters = 10;
constexpr uint32_t kValuesPerThread = 4;

struct DecodedJpegImage {
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t channels = 0;
	std::vector<uint8_t> pixels;
};

struct BenchmarkStats {
	std::vector<double> decode_seconds;
	std::vector<double> upload_seconds;
	std::vector<double> gpu_seconds;
	std::vector<double> readback_seconds;
	double cpu_average_brightness = 0.0;
	double gpu_average_brightness = 0.0;
	uint64_t cpu_total_sum = 0;
	uint64_t gpu_total_sum = 0;
	uint32_t element_count = 0;
};

struct ReducePushConstants {
	uint32_t n;
	uint32_t values_per_thread;
};

bool hasJpegExtension(const fs::path &path)
{
	std::string extension = path.extension().string();
	std::transform(extension.begin(), extension.end(), extension.begin(),
				   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return extension == ".jpg" || extension == ".jpeg";
}

std::vector<uint8_t> readFileBytes(const fs::path &path)
{
	std::ifstream input(path, std::ios::binary);
	rassert(input.good(), 2026032023361400001, path.string());
	input.seekg(0, std::ios::end);
	const std::streamoff file_size = input.tellg();
	rassert(file_size >= 0, 2026032023361400002, path.string(), file_size);
	input.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
	if (!bytes.empty()) {
		input.read(reinterpret_cast<char *>(bytes.data()), file_size);
	}
	rassert(input.good() || input.eof(), 2026032023361400003, path.string());
	return bytes;
}

DecodedJpegImage decodeJpegFromMemory(const std::vector<uint8_t> &compressed)
{
	rassert(!compressed.empty(), 2026032023361400004);

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, compressed.data(), compressed.size());

	const int header_status = jpeg_read_header(&cinfo, TRUE);
	rassert(header_status == JPEG_HEADER_OK, 2026032023361400005, header_status);

	jpeg_start_decompress(&cinfo);

	DecodedJpegImage decoded;
	decoded.width = cinfo.output_width;
	decoded.height = cinfo.output_height;
	decoded.channels = cinfo.output_components;
	rassert(decoded.width > 0 && decoded.height > 0, 2026032023361400006, decoded.width, decoded.height);
	rassert(decoded.channels >= 1 && decoded.channels <= 4, 2026032023361400007, decoded.channels);

	const size_t row_stride = size_t(decoded.width) * decoded.channels;
	decoded.pixels.resize(size_t(decoded.height) * row_stride);
	while (cinfo.output_scanline < cinfo.output_height) {
		JSAMPROW row_pointer = decoded.pixels.data() + size_t(cinfo.output_scanline) * row_stride;
		jpeg_read_scanlines(&cinfo, &row_pointer, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	return decoded;
}

uint64_t sumBytesCpu(const std::vector<uint8_t> &values)
{
	return std::accumulate(values.begin(), values.end(), uint64_t(0));
}

uint32_t divCeilU32(uint32_t value, uint32_t divisor)
{
	rassert(divisor > 0, 2026032023361400008, divisor);
	return (value + divisor - 1) / divisor;
}

template <typename T>
gpu::gpu_mem_32u *launchReductionOnGpu(const gpu::gpu_mem<T> &input,
									   uint32_t input_count,
									   gpu::gpu_mem_32u &scratch_a,
									   gpu::gpu_mem_32u &scratch_b,
									   avk2::KernelSource &kernel,
									   uint64_t input_upper_bound,
									   uint32_t &reduced_count)
{
	rassert(input_count > 0, 2026032023361400009, input_count);
	rassert(input_upper_bound > 0, 2026032214233400001, input_upper_bound);
	uint32_t group_count = divCeilU32(input_count, VK_GROUP_SIZE * kValuesPerThread);
	scratch_a.resizeN(std::max(1u, group_count));
	ReducePushConstants params{input_count, kValuesPerThread};
	kernel.exec(params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, size_t(group_count) * VK_GROUP_SIZE), input, scratch_a);

	uint32_t current_count = group_count;
	uint64_t current_upper_bound = input_upper_bound * VK_GROUP_SIZE * kValuesPerThread;
	gpu::gpu_mem_32u *current = &scratch_a;
	gpu::gpu_mem_32u *next = &scratch_b;
	avk2::KernelSource kernel_u32(avk2::getReduceSumU32ToU32Kernel());
	while (current_count > 1) {
		if (current_upper_bound > std::numeric_limits<uint32_t>::max() / (VK_GROUP_SIZE * kValuesPerThread)) {
			break;
		}
		group_count = divCeilU32(current_count, VK_GROUP_SIZE * kValuesPerThread);
		next->resizeN(std::max(1u, group_count));
		ReducePushConstants next_params{current_count, kValuesPerThread};
		kernel_u32.exec(next_params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, size_t(group_count) * VK_GROUP_SIZE), *current, *next);
		current_count = group_count;
		current_upper_bound *= VK_GROUP_SIZE * kValuesPerThread;
		std::swap(current, next);
	}

	gpu::Context context;
	context.vk()->waitForAllInflightComputeLaunches();
	reduced_count = current_count;
	return current;
}

fs::path findDefaultBenchmarkDir()
{
	fs::path current = fs::current_path();
	for (;;) {
		const fs::path candidate = current / "data/jpeg_benchmark";
		if (fs::exists(candidate)) {
			return candidate;
		}
		if (current == current.root_path()) {
			break;
		}
		current = current.parent_path();
	}
	return {};
}

BenchmarkStats runSingleImageBenchmark(const fs::path &path)
{
	BenchmarkStats result;
	const std::vector<uint8_t> compressed = readFileBytes(path);
	rassert(!compressed.empty(), 2026032023361400010, path.string());

	DecodedJpegImage decoded_once = decodeJpegFromMemory(compressed);
	result.cpu_total_sum = sumBytesCpu(decoded_once.pixels);
	result.element_count = static_cast<uint32_t>(decoded_once.pixels.size());
	rassert(result.element_count > 0, 2026032023361400011, path.string());
	result.cpu_average_brightness = static_cast<double>(result.cpu_total_sum) / result.element_count;

	gpu::gpu_mem_8u decoded_gpu;
	gpu::gpu_mem_32u partial_sums_a;
	gpu::gpu_mem_32u partial_sums_b;
	decoded_gpu.resizeN(decoded_once.pixels.size());

	avk2::KernelSource kernel_u8(avk2::getReduceSumU8ToU32Kernel());

	for (uint32_t iter = 0; iter < kBenchmarkIters; ++iter) {
		timer decode_timer;
		DecodedJpegImage decoded;
		{
			profiling::ScopedRange scope("jpeg cpu decode", 0xFF2E86C1);
			decoded = decodeJpegFromMemory(compressed);
		}
		result.decode_seconds.push_back(decode_timer.elapsed());

		rassert(decoded.width == decoded_once.width, 2026032023361400012, decoded.width, decoded_once.width);
		rassert(decoded.height == decoded_once.height, 2026032023361400013, decoded.height, decoded_once.height);
		rassert(decoded.channels == decoded_once.channels, 2026032023361400014, decoded.channels, decoded_once.channels);
		rassert(decoded.pixels.size() == decoded_once.pixels.size(), 2026032023361400015, decoded.pixels.size(), decoded_once.pixels.size());

		timer upload_timer;
		{
			profiling::ScopedRange scope("jpeg upload", 0xFF239B56);
			decoded_gpu.writeN(decoded.pixels.data(), decoded.pixels.size());
		}
		result.upload_seconds.push_back(upload_timer.elapsed());

		timer gpu_timer;
		gpu::gpu_mem_32u *final_sum_buffer = nullptr;
		uint32_t reduced_count = 0;
		{
			profiling::ScopedRange scope("jpeg gpu reduce", 0xFFC0392B);
			final_sum_buffer = launchReductionOnGpu(decoded_gpu,
													static_cast<uint32_t>(decoded.pixels.size()),
													partial_sums_a,
													partial_sums_b,
													kernel_u8,
													255u,
													reduced_count);
		}
		result.gpu_seconds.push_back(gpu_timer.elapsed());

		timer readback_timer;
		std::vector<uint32_t> readback_values;
		{
			profiling::ScopedRange scope("jpeg readback", 0xFFF39C12);
			readback_values = final_sum_buffer->readVector(reduced_count);
		}
		result.readback_seconds.push_back(readback_timer.elapsed());

		const uint64_t readback_value = std::accumulate(readback_values.begin(), readback_values.end(), uint64_t(0));
		EXPECT_EQ(readback_value, result.cpu_total_sum);
		result.gpu_total_sum = readback_value;
		result.gpu_average_brightness = static_cast<double>(readback_value) / result.element_count;
	}

	return result;
}

std::vector<fs::path> collectBenchmarkImages()
{
	const char *env_dir = std::getenv("GPGPU_VULKAN_JPEG_BENCHMARK_DIR");
	fs::path dir = env_dir ? fs::path(env_dir) : findDefaultBenchmarkDir();
	if (dir.empty() || !fs::exists(dir)) {
		return {};
	}

	std::vector<fs::path> images;
	for (const fs::directory_entry &entry : fs::directory_iterator(dir)) {
		if (!entry.is_regular_file()) {
			continue;
		}
		if (!hasJpegExtension(entry.path())) {
			continue;
		}
		images.push_back(entry.path());
	}
	std::sort(images.begin(), images.end());
	return images;
}

void printBenchmarkStats(const fs::path &path, const BenchmarkStats &stats)
{
	std::cout << "jpeg benchmark: " << path.filename().string()
			  << " compressed_bytes=" << fs::file_size(path)
			  << " cpu_sum=" << stats.cpu_total_sum
			  << " gpu_sum=" << stats.gpu_total_sum
			  << " avg_brightness=" << std::fixed << std::setprecision(6) << stats.gpu_average_brightness
			  << std::endl;
	std::cout << "  cpu jpeg decode median:       " << stats::median(stats.decode_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.decode_seconds) << ")" << std::endl;
	std::cout << "  upload decoded image median:  " << stats::median(stats.upload_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.upload_seconds) << ")" << std::endl;
	std::cout << "  gpu brightness reduce median: " << stats::median(stats.gpu_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.gpu_seconds) << ")" << std::endl;
	std::cout << "  readback median:              " << stats::median(stats.readback_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.readback_seconds) << ")" << std::endl;
}

}

TEST(vulkan, jpegDecodeBenchmark)
{
	std::vector<fs::path> image_paths = collectBenchmarkImages();
	if (image_paths.empty()) {
		GTEST_SKIP() << "No JPEG benchmark images found. Set GPGPU_VULKAN_JPEG_BENCHMARK_DIR or add *.jpg under data/jpeg_benchmark";
	}

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		for (const fs::path &path : image_paths) {
			const std::string scope_name = std::string("jpeg image ") + path.filename().string();
			profiling::ScopedRange scope(scope_name.c_str(), 0xFF8E44AD);
			const BenchmarkStats stats = runSingleImageBenchmark(path);
			EXPECT_NEAR(stats.cpu_average_brightness, stats.gpu_average_brightness, 1e-9);
			printBenchmarkStats(path, stats);
		}
	}

	checkPostInvariants();
}
