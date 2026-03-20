#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
constexpr int kGeneratedJpegQuality = 95;
constexpr unsigned int kGeneratedRestartInterval = 1;

struct DecodedGrayImage {
	uint32_t width = 0;
	uint32_t height = 0;
	std::vector<uint8_t> pixels;
};

struct ReducePushConstants {
	uint32_t n;
	uint32_t values_per_thread;
};

struct GpuJpegDecodePushConstants {
	uint32_t width;
	uint32_t height;
	uint32_t width_blocks;
	uint32_t total_blocks;
};

struct GpuJpegData {
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t width_blocks = 0;
	uint32_t total_blocks = 0;
	std::array<uint32_t, 64> quant_table{};
	std::vector<uint32_t> dc_lut;
	std::vector<uint32_t> ac_lut;
	std::vector<uint32_t> scan_words;
	std::vector<uint32_t> start_positions;
};

struct GpuPreparedImage {
	fs::path source_path;
	fs::path generated_jpeg_path;
	std::vector<uint8_t> generated_jpeg_bytes;
	DecodedGrayImage cpu_reference;
	GpuJpegData gpu_jpeg;
};

struct GpuDecodeBenchmarkStats {
	std::vector<double> upload_seconds;
	std::vector<double> jpeg_decode_seconds;
	std::vector<double> brightness_reduce_seconds;
	std::vector<double> readback_seconds;
	uint64_t cpu_total_sum = 0;
	uint32_t gpu_total_sum = 0;
	uint32_t pixel_count = 0;
	double cpu_average_brightness = 0.0;
	double gpu_average_brightness = 0.0;
};

struct GpuDecodeDiffStats {
	double average_abs_diff = 0.0;
	uint32_t max_abs_diff = 0;
};

struct HuffmanSpec {
	std::array<uint8_t, 16> counts{};
	std::vector<uint8_t> values;
	bool defined = false;
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
	rassert(input.good(), 2026032101384700001, path.string());
	input.seekg(0, std::ios::end);
	const std::streamoff file_size = input.tellg();
	rassert(file_size >= 0, 2026032101384700002, path.string(), file_size);
	input.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
	if (!bytes.empty()) {
		input.read(reinterpret_cast<char *>(bytes.data()), file_size);
	}
	rassert(input.good() || input.eof(), 2026032101384700003, path.string());
	return bytes;
}

void writeFileBytes(const fs::path &path, const std::vector<uint8_t> &bytes)
{
	std::ofstream output(path, std::ios::binary);
	rassert(output.good(), 2026032101384700004, path.string());
	if (!bytes.empty()) {
		output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}
	rassert(output.good(), 2026032101384700005, path.string());
}

DecodedGrayImage decodeJpegToGrayFromMemory(const std::vector<uint8_t> &compressed)
{
	rassert(!compressed.empty(), 2026032101384700006);

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, compressed.data(), compressed.size());
	rassert(jpeg_read_header(&cinfo, TRUE) == JPEG_HEADER_OK, 2026032101384700007);
	cinfo.out_color_space = JCS_GRAYSCALE;
	jpeg_start_decompress(&cinfo);

	DecodedGrayImage decoded;
	decoded.width = cinfo.output_width;
	decoded.height = cinfo.output_height;
	rassert(cinfo.output_components == 1, 2026032101384700008, cinfo.output_components);
	decoded.pixels.resize(size_t(decoded.width) * decoded.height);

	while (cinfo.output_scanline < cinfo.output_height) {
		JSAMPROW row_pointer = decoded.pixels.data() + size_t(cinfo.output_scanline) * decoded.width;
		jpeg_read_scanlines(&cinfo, &row_pointer, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	return decoded;
}

std::vector<uint8_t> encodeGrayJpegToMemory(const DecodedGrayImage &image, int quality, unsigned int restart_interval)
{
	rassert(image.width > 0 && image.height > 0, 2026032101384700009, image.width, image.height);
	rassert(image.pixels.size() == size_t(image.width) * image.height,
			2026032101384700010, image.pixels.size(), image.width, image.height);

	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	unsigned char *outbuffer = nullptr;
	unsigned long outsize = 0;
	jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

	cinfo.image_width = image.width;
	cinfo.image_height = image.height;
	cinfo.input_components = 1;
	cinfo.in_color_space = JCS_GRAYSCALE;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	cinfo.restart_interval = restart_interval;
	jpeg_start_compress(&cinfo, TRUE);

	while (cinfo.next_scanline < cinfo.image_height) {
		JSAMPROW row_pointer = const_cast<JSAMPROW>(image.pixels.data() + size_t(cinfo.next_scanline) * image.width);
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}

	jpeg_finish_compress(&cinfo);
	std::vector<uint8_t> encoded(outbuffer, outbuffer + outsize);
	free(outbuffer);
	jpeg_destroy_compress(&cinfo);
	return encoded;
}

uint64_t sumPixelsCpu(const std::vector<uint8_t> &values)
{
	return std::accumulate(values.begin(), values.end(), uint64_t(0));
}

uint32_t divCeilU32(uint32_t value, uint32_t divisor)
{
	rassert(divisor > 0, 2026032101384700011, divisor);
	return (value + divisor - 1) / divisor;
}

template <typename T>
gpu::gpu_mem_32u *launchReductionOnGpu(const gpu::gpu_mem<T> &input,
									   uint32_t input_count,
									   gpu::gpu_mem_32u &scratch_a,
									   gpu::gpu_mem_32u &scratch_b,
									   avk2::KernelSource &kernel)
{
	rassert(input_count > 0, 2026032101384700012, input_count);
	uint32_t group_count = divCeilU32(input_count, VK_GROUP_SIZE * kValuesPerThread);
	scratch_a.resizeN(std::max(1u, group_count));
	ReducePushConstants params{input_count, kValuesPerThread};
	kernel.exec(params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, size_t(group_count) * VK_GROUP_SIZE), input, scratch_a);

	uint32_t current_count = group_count;
	gpu::gpu_mem_32u *current = &scratch_a;
	gpu::gpu_mem_32u *next = &scratch_b;
	avk2::KernelSource kernel_u32(avk2::getReduceSumU32ToU32Kernel());
	while (current_count > 1) {
		group_count = divCeilU32(current_count, VK_GROUP_SIZE * kValuesPerThread);
		next->resizeN(std::max(1u, group_count));
		ReducePushConstants next_params{current_count, kValuesPerThread};
		kernel_u32.exec(next_params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, size_t(group_count) * VK_GROUP_SIZE), *current, *next);
		current_count = group_count;
		std::swap(current, next);
	}

	gpu::Context context;
	context.vk()->waitForAllInflightComputeLaunches();
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

class ByteReader {
public:
	explicit ByteReader(const std::vector<uint8_t> &data) : data_(data), pos_(0) {}

	size_t pos() const { return pos_; }
	void setPos(size_t pos)
	{
		rassert(pos <= data_.size(), 2026032101384700013, pos, data_.size());
		pos_ = pos;
	}

	bool eof() const { return pos_ >= data_.size(); }

	uint8_t readU8()
	{
		rassert(pos_ < data_.size(), 2026032101384700014, pos_, data_.size());
		return data_[pos_++];
	}

	uint16_t readBE16()
	{
		uint16_t high = readU8();
		uint16_t low = readU8();
		return uint16_t((high << 8) | low);
	}

	void skip(size_t n)
	{
		rassert(pos_ + n <= data_.size(), 2026032101384700015, pos_, n, data_.size());
		pos_ += n;
	}

private:
	const std::vector<uint8_t> &data_;
	size_t pos_;
};

std::vector<uint32_t> buildFullHuffmanLut(const HuffmanSpec &spec)
{
	rassert(spec.defined, 2026032101384700016);
	std::vector<uint32_t> lut(1u << 16u, 0xffffffffu);

	uint32_t code = 0;
	size_t value_index = 0;
	for (uint32_t length = 1; length <= 16; ++length) {
		for (uint32_t i = 0; i < spec.counts[length - 1]; ++i) {
			rassert(value_index < spec.values.size(), 2026032101384700017, value_index, spec.values.size());
			uint32_t symbol = spec.values[value_index++];
			uint32_t fill_begin = code << (16u - length);
			uint32_t fill_end = (code + 1u) << (16u - length);
			uint32_t entry = symbol | (length << 8u);
			for (uint32_t lut_index = fill_begin; lut_index < fill_end; ++lut_index) {
				lut[lut_index] = entry;
			}
			code += 1u;
		}
		code <<= 1u;
	}
	rassert(value_index == spec.values.size(), 2026032101384700018, value_index, spec.values.size());
	return lut;
}

struct ScanPreprocessResult {
	std::vector<uint32_t> words;
	std::vector<uint32_t> start_positions;
};

ScanPreprocessResult preprocessScanData(const std::vector<uint8_t> &scan_data, uint32_t expected_intervals)
{
	std::vector<uint8_t> out(scan_data.size() + scan_data.size() / 3 + 16, uint8_t(0));
	std::vector<uint32_t> start_positions;
	start_positions.reserve(expected_intervals);
	start_positions.push_back(0);

	size_t write_ptr = 0;
	size_t ri = 1;
	for (size_t pos = 0; pos < scan_data.size();) {
		uint8_t byte = scan_data[pos++];
		if (byte != 0xffu) {
			out[write_ptr++] = byte;
			continue;
		}

		rassert(pos < scan_data.size(), 2026032101384700019, scan_data.size());
		uint8_t next = scan_data[pos++];
		if (next == 0x00u) {
			out[write_ptr++] = 0xffu;
		} else if (next >= 0xd0u && next <= 0xd7u) {
			write_ptr = (write_ptr + 3u) & ~size_t(3u);
			start_positions.push_back(static_cast<uint32_t>(write_ptr / 4u));
			ri += 1;
		} else {
			rassert(false, 2026032101384700020, int(next));
		}
	}

	rassert(ri == expected_intervals, 2026032101384700021, ri, expected_intervals);
	size_t padded_bytes = (write_ptr + 3u) & ~size_t(3u);
	std::vector<uint32_t> words(padded_bytes / 4u + 2u, 0u);
	if (padded_bytes > 0) {
		memcpy(words.data(), out.data(), padded_bytes);
	}

	return {words, start_positions};
}

GpuJpegData parseGrayscaleBaselineJpegForGpu(const std::vector<uint8_t> &jpeg_bytes)
{
	ByteReader reader(jpeg_bytes);
	rassert(reader.readU8() == 0xffu && reader.readU8() == 0xd8u, 2026032101384700022);

	uint32_t width = 0;
	uint32_t height = 0;
	uint8_t component_id = 0;
	uint8_t dc_table_selector = 0;
	uint8_t ac_table_selector = 0;
	uint8_t qtable_selector = 0;
	uint32_t restart_interval = 0;
	std::array<std::array<uint32_t, 64>, 4> qtables{};
	std::array<bool, 4> qtable_defined{};
	std::array<HuffmanSpec, 4> huffman_specs{};
	size_t scan_data_offset = 0;
	size_t scan_data_size = 0;
	bool found_sos = false;

	while (!reader.eof()) {
		uint8_t prefix = reader.readU8();
		if (prefix != 0xffu) {
			continue;
		}
		uint8_t marker = reader.readU8();
		while (marker == 0xffu) {
			marker = reader.readU8();
		}

		if (marker == 0xd9u) {
			break;
		}
		if (marker == 0xd8u) {
			continue;
		}

		uint16_t segment_length = reader.readBE16();
		rassert(segment_length >= 2u, 2026032101384700023, segment_length);
		size_t segment_end = reader.pos() + segment_length - 2u;
		rassert(segment_end <= jpeg_bytes.size(), 2026032101384700024, segment_end, jpeg_bytes.size());

		if (marker == 0xc0u) {
			uint8_t precision = reader.readU8();
			height = reader.readBE16();
			width = reader.readBE16();
			uint8_t components = reader.readU8();
			rassert(precision == 8u, 2026032101384700025, precision);
			rassert(components == 1u, 2026032101384700026, components);
			component_id = reader.readU8();
			uint8_t sampling = reader.readU8();
			qtable_selector = reader.readU8();
			rassert((sampling >> 4u) == 1u && (sampling & 0x0fu) == 1u, 2026032101384700027, sampling);
		} else if (marker == 0xdbu) {
			while (reader.pos() < segment_end) {
				uint8_t pq_tq = reader.readU8();
				uint8_t precision = pq_tq >> 4u;
				uint8_t table_id = pq_tq & 0x0fu;
				rassert(precision == 0u, 2026032101384700028, precision);
				rassert(table_id < 4u, 2026032101384700029, table_id);
				for (size_t i = 0; i < 64; ++i) {
					qtables[table_id][i] = reader.readU8();
				}
				qtable_defined[table_id] = true;
			}
		} else if (marker == 0xc4u) {
			while (reader.pos() < segment_end) {
				uint8_t tc_th = reader.readU8();
				uint8_t table_class = tc_th >> 4u;
				uint8_t table_id = tc_th & 0x0fu;
				rassert(table_class <= 1u, 2026032101384700030, table_class);
				rassert(table_id <= 1u, 2026032101384700031, table_id);
				HuffmanSpec &spec = huffman_specs[(table_id << 1u) | table_class];
				uint32_t total_values = 0;
				for (size_t i = 0; i < 16; ++i) {
					spec.counts[i] = reader.readU8();
					total_values += spec.counts[i];
				}
				spec.values.resize(total_values);
				for (uint32_t i = 0; i < total_values; ++i) {
					spec.values[i] = reader.readU8();
				}
				spec.defined = true;
			}
		} else if (marker == 0xddu) {
			restart_interval = reader.readBE16();
		} else if (marker == 0xdau) {
			uint8_t components = reader.readU8();
			rassert(components == 1u, 2026032101384700032, components);
			uint8_t scan_component = reader.readU8();
			uint8_t table_selectors = reader.readU8();
			rassert(scan_component == component_id, 2026032101384700033, int(scan_component), int(component_id));
			dc_table_selector = table_selectors >> 4u;
			ac_table_selector = table_selectors & 0x0fu;
			rassert(reader.readU8() == 0u, 2026032101384700034);
			rassert(reader.readU8() == 63u, 2026032101384700035);
			rassert(reader.readU8() == 0u, 2026032101384700036);

			scan_data_offset = segment_end;
			size_t pos = scan_data_offset;
			while (pos + 1u < jpeg_bytes.size()) {
				if (jpeg_bytes[pos] == 0xffu) {
					uint8_t next = jpeg_bytes[pos + 1u];
					if (next == 0x00u || (next >= 0xd0u && next <= 0xd7u)) {
						pos += 2u;
						continue;
					}
					if (next == 0xd9u) {
						scan_data_size = pos - scan_data_offset;
						found_sos = true;
						break;
					}
				}
				pos += 1u;
			}
			break;
		}

		reader.setPos(segment_end);
	}

	rassert(found_sos, 2026032101384700037);
	rassert(width > 0u && height > 0u, 2026032101384700038, width, height);
	rassert(restart_interval == kGeneratedRestartInterval,
			2026032101384700039, restart_interval, kGeneratedRestartInterval);
	rassert(qtable_selector < 4u && qtable_defined[qtable_selector], 2026032101384700040, qtable_selector);
	rassert(dc_table_selector <= 1u && ac_table_selector <= 1u, 2026032101384700041, dc_table_selector, ac_table_selector);
	rassert(huffman_specs[dc_table_selector << 1u].defined, 2026032101384700042, dc_table_selector);
	rassert(huffman_specs[(ac_table_selector << 1u) | 1u].defined, 2026032101384700043, ac_table_selector);

	uint32_t width_blocks = divCeilU32(width, 8u);
	uint32_t height_blocks = divCeilU32(height, 8u);
	uint32_t total_blocks = width_blocks * height_blocks;
	std::vector<uint8_t> scan_data(jpeg_bytes.begin() + static_cast<ptrdiff_t>(scan_data_offset),
								   jpeg_bytes.begin() + static_cast<ptrdiff_t>(scan_data_offset + scan_data_size));
	ScanPreprocessResult processed = preprocessScanData(scan_data, total_blocks);
	rassert(processed.start_positions.size() == total_blocks,
			2026032101384700044, processed.start_positions.size(), total_blocks);

	GpuJpegData gpu;
	gpu.width = width;
	gpu.height = height;
	gpu.width_blocks = width_blocks;
	gpu.total_blocks = total_blocks;
	gpu.quant_table = qtables[qtable_selector];
	gpu.dc_lut = buildFullHuffmanLut(huffman_specs[dc_table_selector << 1u]);
	gpu.ac_lut = buildFullHuffmanLut(huffman_specs[(ac_table_selector << 1u) | 1u]);
	gpu.scan_words = std::move(processed.words);
	gpu.start_positions = std::move(processed.start_positions);
	return gpu;
}

GpuPreparedImage prepareGpuJpegArtifact(const fs::path &source_path)
{
	const std::vector<uint8_t> source_bytes = readFileBytes(source_path);
	const DecodedGrayImage source_gray = decodeJpegToGrayFromMemory(source_bytes);
	const std::vector<uint8_t> generated_jpeg = encodeGrayJpegToMemory(source_gray, kGeneratedJpegQuality, kGeneratedRestartInterval);

	fs::path out_dir = fs::path("/tmp/gpgpu_vulkan_gpu_jpeg_decode_dataset");
	fs::create_directories(out_dir);
	const fs::path generated_path = out_dir / (source_path.stem().string() + "_gray_rst1_q95.jpg");
	writeFileBytes(generated_path, generated_jpeg);

	GpuPreparedImage prepared;
	prepared.source_path = source_path;
	prepared.generated_jpeg_path = generated_path;
	prepared.generated_jpeg_bytes = generated_jpeg;
	prepared.cpu_reference = decodeJpegToGrayFromMemory(generated_jpeg);
	prepared.gpu_jpeg = parseGrayscaleBaselineJpegForGpu(generated_jpeg);
	return prepared;
}

std::vector<fs::path> collectSourceImages()
{
	const char *env_dir = std::getenv("GPGPU_VULKAN_JPEG_BENCHMARK_DIR");
	fs::path dir = env_dir ? fs::path(env_dir) : findDefaultBenchmarkDir();
	if (dir.empty() || !fs::exists(dir)) {
		return {};
	}

	std::vector<fs::path> images;
	for (const fs::directory_entry &entry : fs::directory_iterator(dir)) {
		if (entry.is_regular_file() && hasJpegExtension(entry.path())) {
			images.push_back(entry.path());
		}
	}
	std::sort(images.begin(), images.end());
	return images;
}

GpuDecodeBenchmarkStats runGpuDecodeBenchmark(const GpuPreparedImage &prepared)
{
	GpuDecodeBenchmarkStats stats;
	stats.pixel_count = static_cast<uint32_t>(prepared.cpu_reference.pixels.size());
	stats.cpu_total_sum = sumPixelsCpu(prepared.cpu_reference.pixels);
	stats.cpu_average_brightness = static_cast<double>(stats.cpu_total_sum) / stats.pixel_count;

	gpu::gpu_mem_32u dc_lut_gpu(prepared.gpu_jpeg.dc_lut.size());
	gpu::gpu_mem_32u ac_lut_gpu(prepared.gpu_jpeg.ac_lut.size());
	gpu::gpu_mem_32u qtable_gpu(prepared.gpu_jpeg.quant_table.size());
	gpu::gpu_mem_32u scan_words_gpu(prepared.gpu_jpeg.scan_words.size());
	gpu::gpu_mem_32u start_positions_gpu(prepared.gpu_jpeg.start_positions.size());
	gpu::gpu_mem_32u decoded_pixels_gpu(stats.pixel_count);
	gpu::gpu_mem_32u partial_sums_a;
	gpu::gpu_mem_32u partial_sums_b;

	avk2::KernelSource jpeg_decode_kernel(avk2::getJpegDecodeGrayscaleKernel());
	avk2::KernelSource reduce_kernel(avk2::getReduceSumU32ToU32Kernel());

	const GpuJpegDecodePushConstants decode_params{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.width_blocks,
		prepared.gpu_jpeg.total_blocks
	};

	for (uint32_t iter = 0; iter < kBenchmarkIters; ++iter) {
		timer upload_timer;
		{
			profiling::ScopedRange scope("jpeg gpu upload compressed", 0xFF1ABC9C);
			dc_lut_gpu.writeN(prepared.gpu_jpeg.dc_lut.data(), prepared.gpu_jpeg.dc_lut.size());
			ac_lut_gpu.writeN(prepared.gpu_jpeg.ac_lut.data(), prepared.gpu_jpeg.ac_lut.size());
			qtable_gpu.writeN(prepared.gpu_jpeg.quant_table.data(), prepared.gpu_jpeg.quant_table.size());
			scan_words_gpu.writeN(prepared.gpu_jpeg.scan_words.data(), prepared.gpu_jpeg.scan_words.size());
			start_positions_gpu.writeN(prepared.gpu_jpeg.start_positions.data(), prepared.gpu_jpeg.start_positions.size());
		}
		stats.upload_seconds.push_back(upload_timer.elapsed());

		timer decode_timer;
		{
			profiling::ScopedRange scope("jpeg gpu decode", 0xFFC0392B);
			jpeg_decode_kernel.exec(decode_params,
									gpu::WorkSize1DTo2D(VK_GROUP_SIZE, prepared.gpu_jpeg.total_blocks),
									dc_lut_gpu, ac_lut_gpu, qtable_gpu, scan_words_gpu, start_positions_gpu, decoded_pixels_gpu);
			gpu::Context context;
			context.vk()->waitForAllInflightComputeLaunches();
		}
		stats.jpeg_decode_seconds.push_back(decode_timer.elapsed());

		timer reduce_timer;
		gpu::gpu_mem_32u *sum_gpu = nullptr;
		{
			profiling::ScopedRange scope("jpeg gpu brightness reduce", 0xFF8E44AD);
			sum_gpu = launchReductionOnGpu(decoded_pixels_gpu, stats.pixel_count, partial_sums_a, partial_sums_b, reduce_kernel);
		}
		stats.brightness_reduce_seconds.push_back(reduce_timer.elapsed());

		timer readback_timer;
		uint32_t gpu_sum = std::numeric_limits<uint32_t>::max();
		{
			profiling::ScopedRange scope("jpeg gpu brightness readback", 0xFFF39C12);
			sum_gpu->readN(&gpu_sum, 1);
		}
		stats.readback_seconds.push_back(readback_timer.elapsed());

		stats.gpu_total_sum = gpu_sum;
		stats.gpu_average_brightness = static_cast<double>(gpu_sum) / stats.pixel_count;
	}

	return stats;
}

GpuDecodeDiffStats compareGpuAndCpuDecode(const GpuPreparedImage &prepared)
{
	gpu::gpu_mem_32u dc_lut_gpu(prepared.gpu_jpeg.dc_lut.size());
	gpu::gpu_mem_32u ac_lut_gpu(prepared.gpu_jpeg.ac_lut.size());
	gpu::gpu_mem_32u qtable_gpu(prepared.gpu_jpeg.quant_table.size());
	gpu::gpu_mem_32u scan_words_gpu(prepared.gpu_jpeg.scan_words.size());
	gpu::gpu_mem_32u start_positions_gpu(prepared.gpu_jpeg.start_positions.size());
	gpu::gpu_mem_32u decoded_pixels_gpu(prepared.cpu_reference.pixels.size());

	dc_lut_gpu.writeN(prepared.gpu_jpeg.dc_lut.data(), prepared.gpu_jpeg.dc_lut.size());
	ac_lut_gpu.writeN(prepared.gpu_jpeg.ac_lut.data(), prepared.gpu_jpeg.ac_lut.size());
	qtable_gpu.writeN(prepared.gpu_jpeg.quant_table.data(), prepared.gpu_jpeg.quant_table.size());
	scan_words_gpu.writeN(prepared.gpu_jpeg.scan_words.data(), prepared.gpu_jpeg.scan_words.size());
	start_positions_gpu.writeN(prepared.gpu_jpeg.start_positions.data(), prepared.gpu_jpeg.start_positions.size());

	avk2::KernelSource jpeg_decode_kernel(avk2::getJpegDecodeGrayscaleKernel());
	const GpuJpegDecodePushConstants decode_params{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.width_blocks,
		prepared.gpu_jpeg.total_blocks
	};

	jpeg_decode_kernel.exec(decode_params,
							gpu::WorkSize1DTo2D(VK_GROUP_SIZE, prepared.gpu_jpeg.total_blocks),
							dc_lut_gpu, ac_lut_gpu, qtable_gpu, scan_words_gpu, start_positions_gpu, decoded_pixels_gpu);
	gpu::Context context;
	context.vk()->waitForAllInflightComputeLaunches();

	std::vector<uint32_t> gpu_pixels_u32 = decoded_pixels_gpu.readVector(prepared.cpu_reference.pixels.size());
	rassert(gpu_pixels_u32.size() == prepared.cpu_reference.pixels.size(),
			2026032101384700045, gpu_pixels_u32.size(), prepared.cpu_reference.pixels.size());

	uint64_t abs_diff_sum = 0;
	uint32_t max_abs_diff = 0;
	for (size_t i = 0; i < gpu_pixels_u32.size(); ++i) {
		uint32_t gpu_value = gpu_pixels_u32[i];
		rassert(gpu_value <= 255u, 2026032101384700046, gpu_value, i);
		uint32_t cpu_value = prepared.cpu_reference.pixels[i];
		uint32_t abs_diff = gpu_value > cpu_value ? gpu_value - cpu_value : cpu_value - gpu_value;
		abs_diff_sum += abs_diff;
		max_abs_diff = std::max(max_abs_diff, abs_diff);
	}

	GpuDecodeDiffStats diff;
	diff.average_abs_diff = static_cast<double>(abs_diff_sum) / gpu_pixels_u32.size();
	diff.max_abs_diff = max_abs_diff;
	return diff;
}

void printGpuDecodeBenchmarkStats(const GpuPreparedImage &prepared, const GpuDecodeBenchmarkStats &stats)
{
	std::cout << "gpu jpeg benchmark: " << prepared.generated_jpeg_path.filename().string()
			  << " source=" << prepared.source_path.filename().string()
			  << " compressed_bytes=" << fs::file_size(prepared.generated_jpeg_path)
			  << " cpu_sum=" << stats.cpu_total_sum
			  << " gpu_sum=" << stats.gpu_total_sum
			  << " avg_brightness=" << std::fixed << std::setprecision(6) << stats.gpu_average_brightness
			  << std::endl;
	std::cout << "  upload compressed median:      " << stats::median(stats.upload_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.upload_seconds) << ")" << std::endl;
	std::cout << "  gpu jpeg decode median:        " << stats::median(stats.jpeg_decode_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.jpeg_decode_seconds) << ")" << std::endl;
	std::cout << "  gpu brightness reduce median:  " << stats::median(stats.brightness_reduce_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.brightness_reduce_seconds) << ")" << std::endl;
	std::cout << "  readback median:               " << stats::median(stats.readback_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.readback_seconds) << ")" << std::endl;
}

void printGpuDecodeDiffStats(const GpuPreparedImage &prepared, const GpuDecodeDiffStats &diff)
{
	std::cout << "gpu jpeg correctness: " << prepared.generated_jpeg_path.filename().string()
			  << " source=" << prepared.source_path.filename().string()
			  << " avg_abs_diff=" << std::fixed << std::setprecision(6) << diff.average_abs_diff
			  << " max_abs_diff=" << diff.max_abs_diff
			  << std::endl;
}

}

TEST(vulkan, jpegDecodeBenchmarkGpu)
{
	std::vector<fs::path> source_paths = collectSourceImages();
	if (source_paths.empty()) {
		GTEST_SKIP() << "No JPEG source images found. Set GPGPU_VULKAN_JPEG_BENCHMARK_DIR or add *.jpg under data/jpeg_benchmark";
	}

	std::vector<GpuPreparedImage> prepared_images;
	prepared_images.reserve(source_paths.size());
	for (const fs::path &source_path : source_paths) {
		prepared_images.push_back(prepareGpuJpegArtifact(source_path));
	}

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		for (const GpuPreparedImage &prepared : prepared_images) {
			const std::string scope_name = std::string("gpu jpeg image ") + prepared.generated_jpeg_path.filename().string();
			profiling::ScopedRange scope(scope_name.c_str(), 0xFF884EA0);
			const GpuDecodeBenchmarkStats stats = runGpuDecodeBenchmark(prepared);
			EXPECT_LE(std::abs(stats.cpu_average_brightness - stats.gpu_average_brightness), 1.0);
			printGpuDecodeBenchmarkStats(prepared, stats);
		}
	}

	checkPostInvariants();
}

TEST(vulkan, jpegDecodeGpuMatchesCpu)
{
	std::vector<fs::path> source_paths = collectSourceImages();
	if (source_paths.empty()) {
		GTEST_SKIP() << "No JPEG source images found. Set GPGPU_VULKAN_JPEG_BENCHMARK_DIR or add *.jpg under data/jpeg_benchmark";
	}

	std::vector<GpuPreparedImage> prepared_images;
	prepared_images.reserve(source_paths.size());
	for (const fs::path &source_path : source_paths) {
		prepared_images.push_back(prepareGpuJpegArtifact(source_path));
	}

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		for (const GpuPreparedImage &prepared : prepared_images) {
			const GpuDecodeDiffStats diff = compareGpuAndCpuDecode(prepared);
			printGpuDecodeDiffStats(prepared, diff);
			EXPECT_LE(diff.average_abs_diff, 2.0);
			EXPECT_LE(diff.max_abs_diff, 20u);
		}
	}

	checkPostInvariants();
}
