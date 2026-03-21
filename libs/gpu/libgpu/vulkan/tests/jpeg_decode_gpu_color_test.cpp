#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
constexpr uint32_t kMaxHuffmanTables = 4;
constexpr uint32_t kHuffmanLutSize = 1u << 16u;
constexpr uint32_t kJpeg420KernelGroupSize = 256;

enum class ChromaSubsampling : uint32_t {
	Yuv420,
	Yuv444
};

enum class Jpeg420Stage2Variant : uint32_t {
	FloatBaseline,
	IntFixedPoint,
	ColumnMetadata
};

enum class JpegBenchmarkSyncMode : uint32_t {
	StageSynchronized,
	IterationSynchronized
};

struct DecodedRgbImage {
	uint32_t width = 0;
	uint32_t height = 0;
	std::vector<uint8_t> pixels;
};

struct ReducePushConstants {
	uint32_t n;
	uint32_t values_per_thread;
};

struct GpuColorDecodePushConstants {
	uint32_t width;
	uint32_t height;
	uint32_t mcu_width;
	uint32_t mcu_height;
	uint32_t mcus_x;
	uint32_t total_mcus;
	uint32_t hmax;
	uint32_t vmax;
	uint32_t comp0_h;
	uint32_t comp0_v;
	uint32_t comp1_h;
	uint32_t comp1_v;
	uint32_t comp2_h;
	uint32_t comp2_v;
	uint32_t comp0_qtable;
	uint32_t comp1_qtable;
	uint32_t comp2_qtable;
	uint32_t comp0_dc_table;
	uint32_t comp0_ac_table;
	uint32_t comp1_dc_table;
	uint32_t comp1_ac_table;
	uint32_t comp2_dc_table;
	uint32_t comp2_ac_table;
};

struct GpuColor420DecodePushConstants {
	uint32_t total_mcus;
	uint32_t y_qtable;
	uint32_t cb_qtable;
	uint32_t cr_qtable;
	uint32_t y_dc_table;
	uint32_t y_ac_table;
	uint32_t cb_dc_table;
	uint32_t cb_ac_table;
	uint32_t cr_dc_table;
	uint32_t cr_ac_table;
};

struct GpuColor420CoeffsToYcbcrPushConstants {
	uint32_t width;
	uint32_t height;
	uint32_t chroma_width;
	uint32_t chroma_height;
	uint32_t mcus_x;
	uint32_t total_mcus;
};

struct Ycbcr420ToRgbPushConstants {
	uint32_t width;
	uint32_t height;
	uint32_t chroma_width;
	uint32_t pixel_count;
};

struct HuffmanSpec {
	std::array<uint8_t, 16> counts{};
	std::vector<uint8_t> values;
	bool defined = false;
};

struct JpegComponentDesc {
	uint8_t id = 0;
	uint8_t h_samp = 0;
	uint8_t v_samp = 0;
	uint8_t qtable_selector = 0;
	uint8_t dc_table_selector = 0;
	uint8_t ac_table_selector = 0;
};

struct GpuColorJpegData {
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t mcu_width = 0;
	uint32_t mcu_height = 0;
	uint32_t mcus_x = 0;
	uint32_t total_mcus = 0;
	uint32_t hmax = 0;
	uint32_t vmax = 0;
	uint32_t chroma_width = 0;
	uint32_t chroma_height = 0;
	std::array<JpegComponentDesc, 3> components{};
	std::vector<uint32_t> quant_tables;
	std::vector<uint32_t> dc_luts;
	std::vector<uint32_t> ac_luts;
	std::vector<uint32_t> scan_words;
	std::vector<uint32_t> start_positions;
};

struct GpuPreparedColorImage {
	fs::path source_path;
	fs::path generated_jpeg_path;
	std::string subsampling_name;
	ChromaSubsampling subsampling = ChromaSubsampling::Yuv420;
	std::vector<uint8_t> generated_jpeg_bytes;
	DecodedRgbImage cpu_reference;
	GpuColorJpegData gpu_jpeg;
};

struct GpuColorDecodeBenchmarkStats {
	std::string stage2_variant_name;
	std::string sync_mode_name;
	std::vector<double> upload_seconds;
	std::vector<double> jpeg_decode_seconds;
	std::vector<double> jpeg_entropy_to_coeffs_seconds;
	std::vector<double> jpeg_coeffs_to_ycbcr_seconds;
	std::vector<double> jpeg_ycbcr_to_rgb_seconds;
	std::vector<double> brightness_reduce_seconds;
	std::vector<double> readback_seconds;
	double all_ac_zero_block_fraction = -1.0;
	uint64_t cpu_total_sum = 0;
	uint64_t gpu_total_sum = 0;
	uint32_t pixel_count = 0;
	double cpu_average_brightness = 0.0;
	double gpu_average_brightness = 0.0;
};

struct GpuColorDecodeDiffStats {
	double average_abs_brightness_diff = 0.0;
	double max_abs_brightness_diff = 0.0;
	double average_abs_channel_diff = 0.0;
	uint32_t max_abs_channel_diff = 0;
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
	rassert(input.good(), 2026032022080100001, path.string());
	input.seekg(0, std::ios::end);
	const std::streamoff file_size = input.tellg();
	rassert(file_size >= 0, 2026032022080100002, path.string(), file_size);
	input.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
	if (!bytes.empty()) {
		input.read(reinterpret_cast<char *>(bytes.data()), file_size);
	}
	rassert(input.good() || input.eof(), 2026032022080100003, path.string());
	return bytes;
}

fs::path localJpegWorkDir()
{
	return fs::current_path() / ".local_data" / "gpgpu_vulkan_gpu_jpeg_color_decode_dataset";
}

void writeFileBytes(const fs::path &path, const std::vector<uint8_t> &bytes)
{
	std::ofstream output(path, std::ios::binary);
	rassert(output.good(), 2026032022080100004, path.string());
	if (!bytes.empty()) {
		output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}
	rassert(output.good(), 2026032022080100005, path.string());
}

DecodedRgbImage decodeJpegToRgbFromMemory(const std::vector<uint8_t> &compressed)
{
	rassert(!compressed.empty(), 2026032022080100006);

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, compressed.data(), compressed.size());
	rassert(jpeg_read_header(&cinfo, TRUE) == JPEG_HEADER_OK, 2026032022080100007);
	cinfo.out_color_space = JCS_RGB;
	jpeg_start_decompress(&cinfo);

	DecodedRgbImage decoded;
	decoded.width = cinfo.output_width;
	decoded.height = cinfo.output_height;
	rassert(cinfo.output_components == 3, 2026032022080100008, cinfo.output_components);
	decoded.pixels.resize(size_t(decoded.width) * decoded.height * 3u);

	while (cinfo.output_scanline < cinfo.output_height) {
		JSAMPROW row_pointer = decoded.pixels.data() + size_t(cinfo.output_scanline) * decoded.width * 3u;
		jpeg_read_scanlines(&cinfo, &row_pointer, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	return decoded;
}

std::string subsamplingName(ChromaSubsampling subsampling)
{
	return subsampling == ChromaSubsampling::Yuv420 ? "420" : "444";
}

std::string normalizeEnvValue(const char *value)
{
	if (value == nullptr) {
		return {};
	}
	std::string normalized = value;
	std::transform(normalized.begin(), normalized.end(), normalized.begin(),
				   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return normalized;
}

Jpeg420Stage2Variant selectedJpeg420Stage2Variant()
{
	const std::string value = normalizeEnvValue(std::getenv("GPGPU_VULKAN_JPEG_420_STAGE2_VARIANT"));
	if (value.empty() || value == "float" || value == "baseline") {
		return Jpeg420Stage2Variant::FloatBaseline;
	}
	if (value == "int" || value == "integer" || value == "fixed") {
		return Jpeg420Stage2Variant::IntFixedPoint;
	}
	if (value == "colmeta" || value == "column_meta" || value == "column-metadata") {
		return Jpeg420Stage2Variant::ColumnMetadata;
	}
	rassert(false, 2026032117405000001, value);
	return Jpeg420Stage2Variant::FloatBaseline;
}

const char *jpeg420Stage2VariantName(Jpeg420Stage2Variant variant)
{
	switch (variant) {
		case Jpeg420Stage2Variant::FloatBaseline:
			return "float";
		case Jpeg420Stage2Variant::IntFixedPoint:
			return "int";
		case Jpeg420Stage2Variant::ColumnMetadata:
			return "colmeta";
	}
	rassert(false, 2026032117405000002);
	return "unknown";
}

JpegBenchmarkSyncMode selectedJpegBenchmarkSyncMode()
{
	const std::string value = normalizeEnvValue(std::getenv("GPGPU_VULKAN_JPEG_BENCH_SYNC_MODE"));
	if (value.empty() || value == "stage" || value == "staged") {
		return JpegBenchmarkSyncMode::StageSynchronized;
	}
	if (value == "iter" || value == "iteration" || value == "throughput") {
		return JpegBenchmarkSyncMode::IterationSynchronized;
	}
	rassert(false, 2026032117405000003, value);
	return JpegBenchmarkSyncMode::StageSynchronized;
}

const char *jpegBenchmarkSyncModeName(JpegBenchmarkSyncMode mode)
{
	switch (mode) {
		case JpegBenchmarkSyncMode::StageSynchronized:
			return "stage";
		case JpegBenchmarkSyncMode::IterationSynchronized:
			return "iteration";
	}
	rassert(false, 2026032117405000004);
	return "unknown";
}

std::vector<uint8_t> encodeRgbJpegToMemory(const DecodedRgbImage &image, int quality,
											 unsigned int restart_interval, ChromaSubsampling subsampling)
{
	rassert(image.width > 0 && image.height > 0, 2026032022080100009, image.width, image.height);
	rassert(image.pixels.size() == size_t(image.width) * image.height * 3u,
			2026032022080100010, image.pixels.size(), image.width, image.height);

	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	unsigned char *outbuffer = nullptr;
	unsigned long outsize = 0;
	jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

	cinfo.image_width = image.width;
	cinfo.image_height = image.height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	cinfo.restart_interval = restart_interval;

	const int y_h = subsampling == ChromaSubsampling::Yuv420 ? 2 : 1;
	const int y_v = subsampling == ChromaSubsampling::Yuv420 ? 2 : 1;
	cinfo.comp_info[0].h_samp_factor = y_h;
	cinfo.comp_info[0].v_samp_factor = y_v;
	cinfo.comp_info[1].h_samp_factor = 1;
	cinfo.comp_info[1].v_samp_factor = 1;
	cinfo.comp_info[2].h_samp_factor = 1;
	cinfo.comp_info[2].v_samp_factor = 1;

	jpeg_start_compress(&cinfo, TRUE);
	while (cinfo.next_scanline < cinfo.image_height) {
		JSAMPROW row_pointer = const_cast<JSAMPROW>(image.pixels.data() + size_t(cinfo.next_scanline) * image.width * 3u);
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
	rassert(divisor > 0, 2026032022080100011, divisor);
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
	rassert(input_count > 0, 2026032022080100012, input_count);
	rassert(input_upper_bound > 0, 2026032214233400002, input_upper_bound);
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

class ByteReader {
public:
	explicit ByteReader(const std::vector<uint8_t> &data) : data_(data), pos_(0) {}

	size_t pos() const { return pos_; }
	void setPos(size_t pos)
	{
		rassert(pos <= data_.size(), 2026032022080100013, pos, data_.size());
		pos_ = pos;
	}

	bool eof() const { return pos_ >= data_.size(); }

	uint8_t readU8()
	{
		rassert(pos_ < data_.size(), 2026032022080100014, pos_, data_.size());
		return data_[pos_++];
	}

	uint16_t readBE16()
	{
		uint16_t high = readU8();
		uint16_t low = readU8();
		return uint16_t((high << 8u) | low);
	}

private:
	const std::vector<uint8_t> &data_;
	size_t pos_;
};

std::vector<uint32_t> buildFullHuffmanLut(const HuffmanSpec &spec)
{
	rassert(spec.defined, 2026032022080100015);
	std::vector<uint32_t> lut(kHuffmanLutSize, 0xffffffffu);

	uint32_t code = 0;
	size_t value_index = 0;
	for (uint32_t length = 1; length <= 16; ++length) {
		for (uint32_t i = 0; i < spec.counts[length - 1]; ++i) {
			rassert(value_index < spec.values.size(), 2026032022080100016, value_index, spec.values.size());
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
	rassert(value_index == spec.values.size(), 2026032022080100017, value_index, spec.values.size());
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
	start_positions.push_back(0u);

	size_t write_ptr = 0;
	size_t intervals_seen = 1;
	for (size_t pos = 0; pos < scan_data.size();) {
		uint8_t byte = scan_data[pos++];
		if (byte != 0xffu) {
			out[write_ptr++] = byte;
			continue;
		}

		rassert(pos < scan_data.size(), 2026032022080100018, pos, scan_data.size());
		uint8_t next = scan_data[pos++];
		if (next == 0x00u) {
			out[write_ptr++] = 0xffu;
		} else if (next >= 0xd0u && next <= 0xd7u) {
			write_ptr = (write_ptr + 3u) & ~size_t(3u);
			start_positions.push_back(static_cast<uint32_t>(write_ptr / 4u));
			intervals_seen += 1;
		} else {
			rassert(false, 2026032022080100019, int(next));
		}
	}

	rassert(intervals_seen == expected_intervals, 2026032022080100020, intervals_seen, expected_intervals);
	size_t padded_bytes = (write_ptr + 3u) & ~size_t(3u);
	std::vector<uint32_t> words(padded_bytes / 4u + 2u, 0u);
	if (padded_bytes > 0) {
		memcpy(words.data(), out.data(), padded_bytes);
	}
	return {words, start_positions};
}

GpuColorJpegData parseColorBaselineJpegForGpu(const std::vector<uint8_t> &jpeg_bytes)
{
	ByteReader reader(jpeg_bytes);
	rassert(reader.readU8() == 0xffu && reader.readU8() == 0xd8u, 2026032022080100021);

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t restart_interval = 0;
	std::array<JpegComponentDesc, 3> components{};
	std::array<std::array<uint32_t, 64>, kMaxHuffmanTables> qtables{};
	std::array<bool, kMaxHuffmanTables> qtable_defined{};
	std::array<HuffmanSpec, kMaxHuffmanTables * 2u> huffman_specs{};
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
		rassert(segment_length >= 2u, 2026032022080100022, segment_length);
		size_t segment_end = reader.pos() + segment_length - 2u;
		rassert(segment_end <= jpeg_bytes.size(), 2026032022080100023, segment_end, jpeg_bytes.size());

		if (marker == 0xc0u) {
			uint8_t precision = reader.readU8();
			height = reader.readBE16();
			width = reader.readBE16();
			uint8_t components_count = reader.readU8();
			rassert(precision == 8u, 2026032022080100024, precision);
			rassert(components_count == 3u, 2026032022080100025, components_count);
			for (uint32_t i = 0; i < 3u; ++i) {
				components[i].id = reader.readU8();
				uint8_t sampling = reader.readU8();
				components[i].h_samp = sampling >> 4u;
				components[i].v_samp = sampling & 0x0fu;
				components[i].qtable_selector = reader.readU8();
				rassert(components[i].h_samp >= 1u && components[i].h_samp <= 2u,
						2026032022080100026, components[i].h_samp);
				rassert(components[i].v_samp >= 1u && components[i].v_samp <= 2u,
						2026032022080100027, components[i].v_samp);
				rassert(components[i].qtable_selector < kMaxHuffmanTables,
						2026032022080100028, components[i].qtable_selector);
			}
		} else if (marker == 0xdbu) {
			while (reader.pos() < segment_end) {
				uint8_t pq_tq = reader.readU8();
				uint8_t precision = pq_tq >> 4u;
				uint8_t table_id = pq_tq & 0x0fu;
				rassert(precision == 0u, 2026032022080100029, precision);
				rassert(table_id < kMaxHuffmanTables, 2026032022080100030, table_id);
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
				rassert(table_class <= 1u, 2026032022080100031, table_class);
				rassert(table_id < kMaxHuffmanTables, 2026032022080100032, table_id);
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
			uint8_t components_count = reader.readU8();
			rassert(components_count == 3u, 2026032022080100033, components_count);
			for (uint32_t i = 0; i < 3u; ++i) {
				uint8_t scan_component_id = reader.readU8();
				uint8_t table_selectors = reader.readU8();
				rassert(components[i].id == scan_component_id, 2026032022080100034, int(components[i].id), int(scan_component_id));
				components[i].dc_table_selector = table_selectors >> 4u;
				components[i].ac_table_selector = table_selectors & 0x0fu;
			}
			rassert(reader.readU8() == 0u, 2026032022080100035);
			rassert(reader.readU8() == 63u, 2026032022080100036);
			rassert(reader.readU8() == 0u, 2026032022080100037);

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

	rassert(found_sos, 2026032022080100038);
	rassert(width > 0u && height > 0u, 2026032022080100039, width, height);
	rassert(restart_interval == kGeneratedRestartInterval,
			2026032022080100040, restart_interval, kGeneratedRestartInterval);

	uint32_t hmax = 1;
	uint32_t vmax = 1;
	for (const JpegComponentDesc &component : components) {
		rassert(qtable_defined[component.qtable_selector], 2026032022080100041, component.qtable_selector);
		rassert(component.dc_table_selector < kMaxHuffmanTables, 2026032022080100042, component.dc_table_selector);
		rassert(component.ac_table_selector < kMaxHuffmanTables, 2026032022080100043, component.ac_table_selector);
		rassert(huffman_specs[component.dc_table_selector << 1u].defined,
				2026032022080100044, component.dc_table_selector);
		rassert(huffman_specs[(component.ac_table_selector << 1u) | 1u].defined,
				2026032022080100045, component.ac_table_selector);
		hmax = std::max(hmax, uint32_t(component.h_samp));
		vmax = std::max(vmax, uint32_t(component.v_samp));
	}

	uint32_t mcu_width = hmax * 8u;
	uint32_t mcu_height = vmax * 8u;
	uint32_t mcus_x = divCeilU32(width, mcu_width);
	uint32_t mcus_y = divCeilU32(height, mcu_height);
	uint32_t total_mcus = mcus_x * mcus_y;

	std::vector<uint8_t> scan_data(jpeg_bytes.begin() + static_cast<ptrdiff_t>(scan_data_offset),
								   jpeg_bytes.begin() + static_cast<ptrdiff_t>(scan_data_offset + scan_data_size));
	ScanPreprocessResult processed = preprocessScanData(scan_data, total_mcus);
	rassert(processed.start_positions.size() == total_mcus,
			2026032022080100046, processed.start_positions.size(), total_mcus);

	GpuColorJpegData gpu;
	gpu.width = width;
	gpu.height = height;
	gpu.mcu_width = mcu_width;
	gpu.mcu_height = mcu_height;
	gpu.mcus_x = mcus_x;
	gpu.total_mcus = total_mcus;
	gpu.hmax = hmax;
	gpu.vmax = vmax;
	gpu.chroma_width = divCeilU32(width, 2u);
	gpu.chroma_height = divCeilU32(height, 2u);
	gpu.components = components;
	gpu.quant_tables.assign(kMaxHuffmanTables * 64u, 0u);
	for (uint32_t table_id = 0; table_id < kMaxHuffmanTables; ++table_id) {
		if (!qtable_defined[table_id]) {
			continue;
		}
		for (uint32_t i = 0; i < 64u; ++i) {
			gpu.quant_tables[table_id * 64u + i] = qtables[table_id][i];
		}
	}

	gpu.dc_luts.assign(kMaxHuffmanTables * kHuffmanLutSize, 0xffffffffu);
	gpu.ac_luts.assign(kMaxHuffmanTables * kHuffmanLutSize, 0xffffffffu);
	for (uint32_t table_id = 0; table_id < kMaxHuffmanTables; ++table_id) {
		const uint32_t dc_spec_index = table_id << 1u;
		const uint32_t ac_spec_index = dc_spec_index | 1u;
		if (huffman_specs[dc_spec_index].defined) {
			std::vector<uint32_t> lut = buildFullHuffmanLut(huffman_specs[dc_spec_index]);
			std::copy(lut.begin(), lut.end(), gpu.dc_luts.begin() + ptrdiff_t(table_id * kHuffmanLutSize));
		}
		if (huffman_specs[ac_spec_index].defined) {
			std::vector<uint32_t> lut = buildFullHuffmanLut(huffman_specs[ac_spec_index]);
			std::copy(lut.begin(), lut.end(), gpu.ac_luts.begin() + ptrdiff_t(table_id * kHuffmanLutSize));
		}
	}
	gpu.scan_words = std::move(processed.words);
	gpu.start_positions = std::move(processed.start_positions);
	return gpu;
}

GpuPreparedColorImage prepareGpuColorJpegArtifact(const fs::path &source_path, ChromaSubsampling subsampling)
{
	const std::vector<uint8_t> source_bytes = readFileBytes(source_path);
	const DecodedRgbImage source_rgb = decodeJpegToRgbFromMemory(source_bytes);
	const std::vector<uint8_t> generated_jpeg = encodeRgbJpegToMemory(source_rgb,
																	 kGeneratedJpegQuality,
																	 kGeneratedRestartInterval,
																	 subsampling);

	const std::string subsampling_name = subsamplingName(subsampling);
	fs::path out_dir = localJpegWorkDir();
	fs::create_directories(out_dir);
	const fs::path generated_path = out_dir / (source_path.stem().string() + "_rgb_" + subsampling_name + "_rst1_q95.jpg");
	writeFileBytes(generated_path, generated_jpeg);

	GpuPreparedColorImage prepared;
	prepared.source_path = source_path;
	prepared.generated_jpeg_path = generated_path;
	prepared.subsampling_name = subsampling_name;
	prepared.subsampling = subsampling;
	prepared.generated_jpeg_bytes = generated_jpeg;
	prepared.cpu_reference = decodeJpegToRgbFromMemory(generated_jpeg);
	prepared.gpu_jpeg = parseColorBaselineJpegForGpu(generated_jpeg);
	return prepared;
}

std::vector<fs::path> collectSourceImages()
{
	const char *env_dir = std::getenv("GPGPU_VULKAN_JPEG_BENCHMARK_DIR");
	fs::path dir = findOrPrepareJpegBenchmarkDir(env_dir);
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

uint8_t unpackChannel(uint32_t packed_rgb, uint32_t channel)
{
	return static_cast<uint8_t>((packed_rgb >> (channel * 8u)) & 0xFFu);
}

double measureAllAcZeroBlockFraction(const GpuPreparedColorImage &prepared)
{
	if (prepared.subsampling != ChromaSubsampling::Yuv420) {
		return -1.0;
	}

	gpu::gpu_mem_32u dc_luts_gpu(prepared.gpu_jpeg.dc_luts.size());
	gpu::gpu_mem_32u ac_luts_gpu(prepared.gpu_jpeg.ac_luts.size());
	gpu::gpu_mem_32u qtables_gpu(prepared.gpu_jpeg.quant_tables.size());
	gpu::gpu_mem_32u scan_words_gpu(prepared.gpu_jpeg.scan_words.size());
	gpu::gpu_mem_32u start_positions_gpu(prepared.gpu_jpeg.start_positions.size());
	gpu::gpu_mem_32i coeffs_gpu(size_t(prepared.gpu_jpeg.total_mcus) * 6u * 64u);
	gpu::gpu_mem_32u all_ac_zero_flags_gpu(size_t(prepared.gpu_jpeg.total_mcus) * 6u);

	dc_luts_gpu.writeN(prepared.gpu_jpeg.dc_luts.data(), prepared.gpu_jpeg.dc_luts.size());
	ac_luts_gpu.writeN(prepared.gpu_jpeg.ac_luts.data(), prepared.gpu_jpeg.ac_luts.size());
	qtables_gpu.writeN(prepared.gpu_jpeg.quant_tables.data(), prepared.gpu_jpeg.quant_tables.size());
	scan_words_gpu.writeN(prepared.gpu_jpeg.scan_words.data(), prepared.gpu_jpeg.scan_words.size());
	start_positions_gpu.writeN(prepared.gpu_jpeg.start_positions.data(), prepared.gpu_jpeg.start_positions.size());

	avk2::KernelSource jpeg_decode_kernel(avk2::getJpegDecodeColor420ToCoeffsKernel());
	const GpuColor420DecodePushConstants decode_params_420{
		prepared.gpu_jpeg.total_mcus,
		prepared.gpu_jpeg.components[0].qtable_selector,
		prepared.gpu_jpeg.components[1].qtable_selector,
		prepared.gpu_jpeg.components[2].qtable_selector,
		prepared.gpu_jpeg.components[0].dc_table_selector,
		prepared.gpu_jpeg.components[0].ac_table_selector,
		prepared.gpu_jpeg.components[1].dc_table_selector,
		prepared.gpu_jpeg.components[1].ac_table_selector,
		prepared.gpu_jpeg.components[2].dc_table_selector,
		prepared.gpu_jpeg.components[2].ac_table_selector
	};
	jpeg_decode_kernel.exec(decode_params_420,
							gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
							dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu);
	gpu::Context context;
	context.vk()->waitForAllInflightComputeLaunches();

	const std::vector<int32_t> coeffs_cpu = coeffs_gpu.readVector();
	const size_t blocks_count = size_t(prepared.gpu_jpeg.total_mcus) * 6u;
	size_t all_ac_zero_blocks = 0;
	for (size_t block = 0; block < blocks_count; ++block) {
		const size_t base = block * 64u;
		bool all_ac_zero = true;
		for (size_t i = 1; i < 64; ++i) {
			if (coeffs_cpu[base + i] != 0) {
				all_ac_zero = false;
				break;
			}
		}
		all_ac_zero_blocks += size_t(all_ac_zero);
	}
	return blocks_count == 0 ? 0.0 : static_cast<double>(all_ac_zero_blocks) / static_cast<double>(blocks_count);
}

GpuColorDecodeBenchmarkStats runGpuColorDecodeBenchmark(const GpuPreparedColorImage &prepared)
{
	GpuColorDecodeBenchmarkStats stats;
	stats.pixel_count = prepared.cpu_reference.width * prepared.cpu_reference.height;
	stats.cpu_total_sum = sumPixelsCpu(prepared.cpu_reference.pixels);
	stats.cpu_average_brightness = static_cast<double>(stats.cpu_total_sum) / (stats.pixel_count * 3u);
	stats.all_ac_zero_block_fraction = measureAllAcZeroBlockFraction(prepared);

	gpu::gpu_mem_32u dc_luts_gpu(prepared.gpu_jpeg.dc_luts.size());
	gpu::gpu_mem_32u ac_luts_gpu(prepared.gpu_jpeg.ac_luts.size());
	gpu::gpu_mem_32u qtables_gpu(prepared.gpu_jpeg.quant_tables.size());
	gpu::gpu_mem_32u scan_words_gpu(prepared.gpu_jpeg.scan_words.size());
	gpu::gpu_mem_32u start_positions_gpu(prepared.gpu_jpeg.start_positions.size());
	gpu::gpu_mem_32i coeffs_gpu;
	gpu::gpu_mem_32u all_ac_zero_flags_gpu;
	gpu::gpu_mem_32u column_meta_gpu;
	gpu::gpu_mem_32u y_plane_gpu;
	gpu::gpu_mem_32u cb_plane_gpu;
	gpu::gpu_mem_32u cr_plane_gpu;
	gpu::gpu_mem_32u decoded_pixels_gpu(stats.pixel_count);
	gpu::gpu_mem_32u partial_sums_a;
	gpu::gpu_mem_32u partial_sums_b;

	const bool use_420_fast_path = prepared.subsampling == ChromaSubsampling::Yuv420;
	const Jpeg420Stage2Variant stage2_variant = selectedJpeg420Stage2Variant();
	const JpegBenchmarkSyncMode sync_mode = selectedJpegBenchmarkSyncMode();
	const bool use_420_colmeta = use_420_fast_path && stage2_variant == Jpeg420Stage2Variant::ColumnMetadata;
	const bool use_420_int = use_420_fast_path && stage2_variant == Jpeg420Stage2Variant::IntFixedPoint;
	stats.stage2_variant_name = jpeg420Stage2VariantName(stage2_variant);
	stats.sync_mode_name = jpegBenchmarkSyncModeName(sync_mode);
	avk2::KernelSource jpeg_decode_kernel(use_420_fast_path
											 ? (use_420_colmeta ? avk2::getJpegDecodeColor420ToCoeffsColmetaKernel()
																: avk2::getJpegDecodeColor420ToCoeffsKernel())
											 : avk2::getJpegDecodeColorKernel());
	avk2::KernelSource coeffs_to_ycbcr_kernel(use_420_colmeta
												 ? avk2::getJpegDecodeColor420CoeffsToYcbcrColmetaKernel()
												 : (use_420_int ? avk2::getJpegDecodeColor420CoeffsToYcbcrIntKernel()
																: avk2::getJpegDecodeColor420CoeffsToYcbcrKernel()));
	avk2::KernelSource ycbcr_to_rgb_kernel(avk2::getYcbcr420ToRgbKernel());
	avk2::KernelSource reduce_kernel(avk2::getReduceSumPackedRgb8ToU32Kernel());

	const GpuColorDecodePushConstants decode_params{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.mcu_width,
		prepared.gpu_jpeg.mcu_height,
		prepared.gpu_jpeg.mcus_x,
		prepared.gpu_jpeg.total_mcus,
		prepared.gpu_jpeg.hmax,
		prepared.gpu_jpeg.vmax,
		prepared.gpu_jpeg.components[0].h_samp,
		prepared.gpu_jpeg.components[0].v_samp,
		prepared.gpu_jpeg.components[1].h_samp,
		prepared.gpu_jpeg.components[1].v_samp,
		prepared.gpu_jpeg.components[2].h_samp,
		prepared.gpu_jpeg.components[2].v_samp,
		prepared.gpu_jpeg.components[0].qtable_selector,
		prepared.gpu_jpeg.components[1].qtable_selector,
		prepared.gpu_jpeg.components[2].qtable_selector,
		prepared.gpu_jpeg.components[0].dc_table_selector,
		prepared.gpu_jpeg.components[0].ac_table_selector,
		prepared.gpu_jpeg.components[1].dc_table_selector,
		prepared.gpu_jpeg.components[1].ac_table_selector,
		prepared.gpu_jpeg.components[2].dc_table_selector,
		prepared.gpu_jpeg.components[2].ac_table_selector
	};
	const GpuColor420DecodePushConstants decode_params_420{
		prepared.gpu_jpeg.total_mcus,
		prepared.gpu_jpeg.components[0].qtable_selector,
		prepared.gpu_jpeg.components[1].qtable_selector,
		prepared.gpu_jpeg.components[2].qtable_selector,
		prepared.gpu_jpeg.components[0].dc_table_selector,
		prepared.gpu_jpeg.components[0].ac_table_selector,
		prepared.gpu_jpeg.components[1].dc_table_selector,
		prepared.gpu_jpeg.components[1].ac_table_selector,
		prepared.gpu_jpeg.components[2].dc_table_selector,
		prepared.gpu_jpeg.components[2].ac_table_selector
	};
	const GpuColor420CoeffsToYcbcrPushConstants coeffs_to_ycbcr_params_420{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.chroma_width,
		prepared.gpu_jpeg.chroma_height,
		prepared.gpu_jpeg.mcus_x,
		prepared.gpu_jpeg.total_mcus
	};
	const Ycbcr420ToRgbPushConstants convert_params_420{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.chroma_width,
		stats.pixel_count
	};
	coeffs_gpu.resizeN(size_t(prepared.gpu_jpeg.total_mcus) * 6u * 64u);
	all_ac_zero_flags_gpu.resizeN(size_t(prepared.gpu_jpeg.total_mcus) * 6u);
	column_meta_gpu.resizeN(size_t(prepared.gpu_jpeg.total_mcus) * 6u);
	y_plane_gpu.resizeN(stats.pixel_count);
	cb_plane_gpu.resizeN(size_t(prepared.gpu_jpeg.chroma_width) * prepared.gpu_jpeg.chroma_height);
	cr_plane_gpu.resizeN(size_t(prepared.gpu_jpeg.chroma_width) * prepared.gpu_jpeg.chroma_height);

	for (uint32_t iter = 0; iter < kBenchmarkIters; ++iter) {
		timer upload_timer;
		{
			profiling::ScopedRange scope("jpeg gpu upload compressed color", 0xFF1ABC9C);
			dc_luts_gpu.writeN(prepared.gpu_jpeg.dc_luts.data(), prepared.gpu_jpeg.dc_luts.size());
			ac_luts_gpu.writeN(prepared.gpu_jpeg.ac_luts.data(), prepared.gpu_jpeg.ac_luts.size());
			qtables_gpu.writeN(prepared.gpu_jpeg.quant_tables.data(), prepared.gpu_jpeg.quant_tables.size());
			scan_words_gpu.writeN(prepared.gpu_jpeg.scan_words.data(), prepared.gpu_jpeg.scan_words.size());
			start_positions_gpu.writeN(prepared.gpu_jpeg.start_positions.data(), prepared.gpu_jpeg.start_positions.size());
		}
		stats.upload_seconds.push_back(upload_timer.elapsed());

		timer decode_timer;
		{
			profiling::ScopedRange scope("jpeg gpu decode color", 0xFFC0392B);
			if (use_420_fast_path) {
				gpu::Context context;
				if (sync_mode == JpegBenchmarkSyncMode::StageSynchronized) {
					timer entropy_to_coeffs_timer;
					if (use_420_colmeta) {
						jpeg_decode_kernel.exec(decode_params_420,
												gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
												dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu);
					} else {
						jpeg_decode_kernel.exec(decode_params_420,
												gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
												dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu);
					}
					context.vk()->waitForAllInflightComputeLaunches();
					stats.jpeg_entropy_to_coeffs_seconds.push_back(entropy_to_coeffs_timer.elapsed());

					timer coeffs_to_ycbcr_timer;
					if (use_420_colmeta) {
						coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
													gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
													coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
					} else {
						coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
													gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
													coeffs_gpu, all_ac_zero_flags_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
					}
					context.vk()->waitForAllInflightComputeLaunches();
					stats.jpeg_coeffs_to_ycbcr_seconds.push_back(coeffs_to_ycbcr_timer.elapsed());

					timer ycbcr_to_rgb_timer;
					ycbcr_to_rgb_kernel.exec(convert_params_420,
											gpu::WorkSize1DTo2D(VK_GROUP_SIZE, stats.pixel_count),
											y_plane_gpu, cb_plane_gpu, cr_plane_gpu, decoded_pixels_gpu);
					context.vk()->waitForAllInflightComputeLaunches();
					stats.jpeg_ycbcr_to_rgb_seconds.push_back(ycbcr_to_rgb_timer.elapsed());
				} else {
					if (use_420_colmeta) {
						jpeg_decode_kernel.exec(decode_params_420,
												gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
												dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu);
						coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
													gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
													coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
					} else {
						jpeg_decode_kernel.exec(decode_params_420,
												gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
												dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu);
						coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
													gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
													coeffs_gpu, all_ac_zero_flags_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
					}
					ycbcr_to_rgb_kernel.exec(convert_params_420,
											gpu::WorkSize1DTo2D(VK_GROUP_SIZE, stats.pixel_count),
											y_plane_gpu, cb_plane_gpu, cr_plane_gpu, decoded_pixels_gpu);
					context.vk()->waitForAllInflightComputeLaunches();
				}
			} else {
				jpeg_decode_kernel.exec(decode_params,
										gpu::WorkSize1DTo2D(VK_GROUP_SIZE, prepared.gpu_jpeg.total_mcus),
										dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, decoded_pixels_gpu);
				gpu::Context context;
				context.vk()->waitForAllInflightComputeLaunches();
			}
		}
		stats.jpeg_decode_seconds.push_back(decode_timer.elapsed());

		timer reduce_timer;
		gpu::gpu_mem_32u *sum_gpu = nullptr;
		uint32_t reduced_count = 0;
		{
			profiling::ScopedRange scope("jpeg gpu brightness reduce color", 0xFF8E44AD);
			sum_gpu = launchReductionOnGpu(decoded_pixels_gpu,
										 stats.pixel_count,
										 partial_sums_a,
										 partial_sums_b,
										 reduce_kernel,
										 255u * 3u,
										 reduced_count);
		}
		stats.brightness_reduce_seconds.push_back(reduce_timer.elapsed());

		timer readback_timer;
		std::vector<uint32_t> gpu_sums;
		{
			profiling::ScopedRange scope("jpeg gpu brightness readback color", 0xFFF39C12);
			gpu_sums = sum_gpu->readVector(reduced_count);
		}
		stats.readback_seconds.push_back(readback_timer.elapsed());

		const uint64_t gpu_sum = std::accumulate(gpu_sums.begin(), gpu_sums.end(), uint64_t(0));
		stats.gpu_total_sum = gpu_sum;
		stats.gpu_average_brightness = static_cast<double>(gpu_sum) / (stats.pixel_count * 3u);
	}

	return stats;
}

GpuColorDecodeDiffStats compareGpuAndCpuDecode(const GpuPreparedColorImage &prepared)
{
	gpu::gpu_mem_32u dc_luts_gpu(prepared.gpu_jpeg.dc_luts.size());
	gpu::gpu_mem_32u ac_luts_gpu(prepared.gpu_jpeg.ac_luts.size());
	gpu::gpu_mem_32u qtables_gpu(prepared.gpu_jpeg.quant_tables.size());
	gpu::gpu_mem_32u scan_words_gpu(prepared.gpu_jpeg.scan_words.size());
	gpu::gpu_mem_32u start_positions_gpu(prepared.gpu_jpeg.start_positions.size());
	gpu::gpu_mem_32i coeffs_gpu;
	gpu::gpu_mem_32u all_ac_zero_flags_gpu;
	gpu::gpu_mem_32u y_plane_gpu;
	gpu::gpu_mem_32u cb_plane_gpu;
	gpu::gpu_mem_32u cr_plane_gpu;
	gpu::gpu_mem_32u decoded_pixels_gpu(prepared.cpu_reference.width * prepared.cpu_reference.height);

	dc_luts_gpu.writeN(prepared.gpu_jpeg.dc_luts.data(), prepared.gpu_jpeg.dc_luts.size());
	ac_luts_gpu.writeN(prepared.gpu_jpeg.ac_luts.data(), prepared.gpu_jpeg.ac_luts.size());
	qtables_gpu.writeN(prepared.gpu_jpeg.quant_tables.data(), prepared.gpu_jpeg.quant_tables.size());
	scan_words_gpu.writeN(prepared.gpu_jpeg.scan_words.data(), prepared.gpu_jpeg.scan_words.size());
	start_positions_gpu.writeN(prepared.gpu_jpeg.start_positions.data(), prepared.gpu_jpeg.start_positions.size());

	const bool use_420_fast_path = prepared.subsampling == ChromaSubsampling::Yuv420;
	const Jpeg420Stage2Variant stage2_variant = selectedJpeg420Stage2Variant();
	const bool use_420_colmeta = use_420_fast_path && stage2_variant == Jpeg420Stage2Variant::ColumnMetadata;
	const bool use_420_int = use_420_fast_path && stage2_variant == Jpeg420Stage2Variant::IntFixedPoint;
	avk2::KernelSource jpeg_decode_kernel(use_420_fast_path
											 ? (use_420_colmeta ? avk2::getJpegDecodeColor420ToCoeffsColmetaKernel()
																: avk2::getJpegDecodeColor420ToCoeffsKernel())
											 : avk2::getJpegDecodeColorKernel());
	avk2::KernelSource coeffs_to_ycbcr_kernel(use_420_colmeta
												 ? avk2::getJpegDecodeColor420CoeffsToYcbcrColmetaKernel()
												 : (use_420_int ? avk2::getJpegDecodeColor420CoeffsToYcbcrIntKernel()
																: avk2::getJpegDecodeColor420CoeffsToYcbcrKernel()));
	avk2::KernelSource ycbcr_to_rgb_kernel(avk2::getYcbcr420ToRgbKernel());
	const GpuColorDecodePushConstants decode_params{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.mcu_width,
		prepared.gpu_jpeg.mcu_height,
		prepared.gpu_jpeg.mcus_x,
		prepared.gpu_jpeg.total_mcus,
		prepared.gpu_jpeg.hmax,
		prepared.gpu_jpeg.vmax,
		prepared.gpu_jpeg.components[0].h_samp,
		prepared.gpu_jpeg.components[0].v_samp,
		prepared.gpu_jpeg.components[1].h_samp,
		prepared.gpu_jpeg.components[1].v_samp,
		prepared.gpu_jpeg.components[2].h_samp,
		prepared.gpu_jpeg.components[2].v_samp,
		prepared.gpu_jpeg.components[0].qtable_selector,
		prepared.gpu_jpeg.components[1].qtable_selector,
		prepared.gpu_jpeg.components[2].qtable_selector,
		prepared.gpu_jpeg.components[0].dc_table_selector,
		prepared.gpu_jpeg.components[0].ac_table_selector,
		prepared.gpu_jpeg.components[1].dc_table_selector,
		prepared.gpu_jpeg.components[1].ac_table_selector,
		prepared.gpu_jpeg.components[2].dc_table_selector,
		prepared.gpu_jpeg.components[2].ac_table_selector
	};
	const GpuColor420DecodePushConstants decode_params_420{
		prepared.gpu_jpeg.total_mcus,
		prepared.gpu_jpeg.components[0].qtable_selector,
		prepared.gpu_jpeg.components[1].qtable_selector,
		prepared.gpu_jpeg.components[2].qtable_selector,
		prepared.gpu_jpeg.components[0].dc_table_selector,
		prepared.gpu_jpeg.components[0].ac_table_selector,
		prepared.gpu_jpeg.components[1].dc_table_selector,
		prepared.gpu_jpeg.components[1].ac_table_selector,
		prepared.gpu_jpeg.components[2].dc_table_selector,
		prepared.gpu_jpeg.components[2].ac_table_selector
	};
	const GpuColor420CoeffsToYcbcrPushConstants coeffs_to_ycbcr_params_420{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.chroma_width,
		prepared.gpu_jpeg.chroma_height,
		prepared.gpu_jpeg.mcus_x,
		prepared.gpu_jpeg.total_mcus
	};
	const Ycbcr420ToRgbPushConstants convert_params_420{
		prepared.gpu_jpeg.width,
		prepared.gpu_jpeg.height,
		prepared.gpu_jpeg.chroma_width,
		prepared.gpu_jpeg.width * prepared.gpu_jpeg.height
	};
	coeffs_gpu.resizeN(size_t(prepared.gpu_jpeg.total_mcus) * 6u * 64u);
	all_ac_zero_flags_gpu.resizeN(size_t(prepared.gpu_jpeg.total_mcus) * 6u);
	gpu::gpu_mem_32u column_meta_gpu(size_t(prepared.gpu_jpeg.total_mcus) * 6u);
	y_plane_gpu.resizeN(size_t(prepared.gpu_jpeg.width) * prepared.gpu_jpeg.height);
	cb_plane_gpu.resizeN(size_t(prepared.gpu_jpeg.chroma_width) * prepared.gpu_jpeg.chroma_height);
	cr_plane_gpu.resizeN(size_t(prepared.gpu_jpeg.chroma_width) * prepared.gpu_jpeg.chroma_height);

	if (use_420_fast_path) {
		if (use_420_colmeta) {
			jpeg_decode_kernel.exec(decode_params_420,
									gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
									dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu);
			coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
									gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
									coeffs_gpu, all_ac_zero_flags_gpu, column_meta_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
		} else {
			jpeg_decode_kernel.exec(decode_params_420,
									gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
									dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, coeffs_gpu, all_ac_zero_flags_gpu);
			coeffs_to_ycbcr_kernel.exec(coeffs_to_ycbcr_params_420,
									gpu::WorkSize1DTo2D(kJpeg420KernelGroupSize, prepared.gpu_jpeg.total_mcus),
									coeffs_gpu, all_ac_zero_flags_gpu, y_plane_gpu, cb_plane_gpu, cr_plane_gpu);
		}
		ycbcr_to_rgb_kernel.exec(convert_params_420,
								gpu::WorkSize1DTo2D(VK_GROUP_SIZE, prepared.gpu_jpeg.width * prepared.gpu_jpeg.height),
								y_plane_gpu, cb_plane_gpu, cr_plane_gpu, decoded_pixels_gpu);
	} else {
		jpeg_decode_kernel.exec(decode_params,
								gpu::WorkSize1DTo2D(VK_GROUP_SIZE, prepared.gpu_jpeg.total_mcus),
								dc_luts_gpu, ac_luts_gpu, qtables_gpu, scan_words_gpu, start_positions_gpu, decoded_pixels_gpu);
	}
	gpu::Context context;
	context.vk()->waitForAllInflightComputeLaunches();

	std::vector<uint32_t> gpu_pixels_packed = decoded_pixels_gpu.readVector(prepared.cpu_reference.width * prepared.cpu_reference.height);
	rassert(gpu_pixels_packed.size() * 3u == prepared.cpu_reference.pixels.size(),
			2026032022080100047, gpu_pixels_packed.size(), prepared.cpu_reference.pixels.size());

	double brightness_abs_diff_sum = 0.0;
	double max_abs_brightness_diff = 0.0;
	uint64_t channel_abs_diff_sum = 0;
	uint32_t max_abs_channel_diff = 0;
	for (size_t pixel_index = 0; pixel_index < gpu_pixels_packed.size(); ++pixel_index) {
		uint32_t packed = gpu_pixels_packed[pixel_index];
		uint32_t cpu_offset = static_cast<uint32_t>(pixel_index * 3u);
		double cpu_brightness = 0.0;
		double gpu_brightness = 0.0;
		for (uint32_t channel = 0; channel < 3u; ++channel) {
			uint32_t gpu_value = unpackChannel(packed, channel);
			uint32_t cpu_value = prepared.cpu_reference.pixels[cpu_offset + channel];
			uint32_t abs_diff = gpu_value > cpu_value ? gpu_value - cpu_value : cpu_value - gpu_value;
			channel_abs_diff_sum += abs_diff;
			max_abs_channel_diff = std::max(max_abs_channel_diff, abs_diff);
			cpu_brightness += cpu_value;
			gpu_brightness += gpu_value;
		}
		cpu_brightness /= 3.0;
		gpu_brightness /= 3.0;
		double brightness_abs_diff = std::abs(cpu_brightness - gpu_brightness);
		brightness_abs_diff_sum += brightness_abs_diff;
		max_abs_brightness_diff = std::max(max_abs_brightness_diff, brightness_abs_diff);
	}

	GpuColorDecodeDiffStats diff;
	diff.average_abs_brightness_diff = brightness_abs_diff_sum / gpu_pixels_packed.size();
	diff.max_abs_brightness_diff = max_abs_brightness_diff;
	diff.average_abs_channel_diff = static_cast<double>(channel_abs_diff_sum) / prepared.cpu_reference.pixels.size();
	diff.max_abs_channel_diff = max_abs_channel_diff;
	return diff;
}

void printGpuColorDecodeBenchmarkStats(const GpuPreparedColorImage &prepared, const GpuColorDecodeBenchmarkStats &stats)
{
	std::cout << "gpu color jpeg benchmark: " << prepared.generated_jpeg_path.filename().string()
			  << " source=" << prepared.source_path.filename().string()
			  << " subsampling=" << prepared.subsampling_name
			  << " stage2_variant=" << stats.stage2_variant_name
			  << " sync_mode=" << stats.sync_mode_name
			  << " compressed_bytes=" << fs::file_size(prepared.generated_jpeg_path)
			  << " all_ac_zero_blocks=" << std::fixed << std::setprecision(4) << stats.all_ac_zero_block_fraction
			  << " cpu_sum=" << stats.cpu_total_sum
			  << " gpu_sum=" << stats.gpu_total_sum
			  << " avg_brightness=" << std::fixed << std::setprecision(6) << stats.gpu_average_brightness
			  << std::endl;
	std::cout << "  upload compressed median:      " << stats::median(stats.upload_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.upload_seconds) << ")" << std::endl;
	std::cout << "  gpu jpeg decode median:        " << stats::median(stats.jpeg_decode_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.jpeg_decode_seconds) << ")" << std::endl;
	if (!stats.jpeg_entropy_to_coeffs_seconds.empty()) {
		std::cout << "    entropy -> coeffs median:    " << stats::median(stats.jpeg_entropy_to_coeffs_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(stats.jpeg_entropy_to_coeffs_seconds) << ")" << std::endl;
		std::cout << "    coeffs -> ycbcr median:      " << stats::median(stats.jpeg_coeffs_to_ycbcr_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(stats.jpeg_coeffs_to_ycbcr_seconds) << ")" << std::endl;
		std::cout << "    ycbcr -> rgb median:         " << stats::median(stats.jpeg_ycbcr_to_rgb_seconds) * 1000.0 << " ms"
				  << " (" << stats::valuesStatsLine(stats.jpeg_ycbcr_to_rgb_seconds) << ")" << std::endl;
	}
	std::cout << "  gpu brightness reduce median:  " << stats::median(stats.brightness_reduce_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.brightness_reduce_seconds) << ")" << std::endl;
	std::cout << "  readback median:               " << stats::median(stats.readback_seconds) * 1000.0 << " ms"
			  << " (" << stats::valuesStatsLine(stats.readback_seconds) << ")" << std::endl;
}

void printGpuColorDecodeDiffStats(const GpuPreparedColorImage &prepared, const GpuColorDecodeDiffStats &diff)
{
	std::cout << "gpu color jpeg correctness: " << prepared.generated_jpeg_path.filename().string()
			  << " source=" << prepared.source_path.filename().string()
			  << " subsampling=" << prepared.subsampling_name
			  << " avg_abs_brightness_diff=" << std::fixed << std::setprecision(6) << diff.average_abs_brightness_diff
			  << " max_abs_brightness_diff=" << diff.max_abs_brightness_diff
			  << " avg_abs_channel_diff=" << diff.average_abs_channel_diff
			  << " max_abs_channel_diff=" << diff.max_abs_channel_diff
			  << std::endl;
}

}

TEST(vulkan, jpegDecodeBenchmarkGpuColor)
{
	std::vector<fs::path> source_paths = collectSourceImages();
	if (source_paths.empty()) {
		GTEST_SKIP() << "No JPEG source images found. Set GPGPU_VULKAN_JPEG_BENCHMARK_DIR or allow lazy download into .local_data/gpgpu_jpeg_benchmark_real";
	}

	std::vector<GpuPreparedColorImage> prepared_images;
	prepared_images.reserve(source_paths.size());
	for (const fs::path &source_path : source_paths) {
		prepared_images.push_back(prepareGpuColorJpegArtifact(source_path, ChromaSubsampling::Yuv420));
	}

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		for (const GpuPreparedColorImage &prepared : prepared_images) {
			const std::string scope_name = std::string("gpu color jpeg image ") + prepared.generated_jpeg_path.filename().string();
			profiling::ScopedRange scope(scope_name.c_str(), 0xFF884EA0);
			const GpuColorDecodeBenchmarkStats stats = runGpuColorDecodeBenchmark(prepared);
			EXPECT_LE(std::abs(stats.cpu_average_brightness - stats.gpu_average_brightness), 2.0);
			printGpuColorDecodeBenchmarkStats(prepared, stats);
		}
	}

	checkPostInvariants();
}

TEST(vulkan, jpegDecodeGpuColorMatchesCpu)
{
	std::vector<fs::path> source_paths = collectSourceImages();
	if (source_paths.empty()) {
		GTEST_SKIP() << "No JPEG source images found. Set GPGPU_VULKAN_JPEG_BENCHMARK_DIR or allow lazy download into .local_data/gpgpu_jpeg_benchmark_real";
	}

	std::vector<GpuPreparedColorImage> prepared_images;
	prepared_images.reserve(source_paths.size() + 1u);
	for (const fs::path &source_path : source_paths) {
		prepared_images.push_back(prepareGpuColorJpegArtifact(source_path, ChromaSubsampling::Yuv420));
	}
	prepared_images.push_back(prepareGpuColorJpegArtifact(source_paths.front(), ChromaSubsampling::Yuv444));

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		for (const GpuPreparedColorImage &prepared : prepared_images) {
			const GpuColorDecodeDiffStats diff = compareGpuAndCpuDecode(prepared);
			printGpuColorDecodeDiffStats(prepared, diff);
			EXPECT_LE(diff.average_abs_brightness_diff, 2.0);
			EXPECT_LE(diff.max_abs_brightness_diff, 20.0);
			EXPECT_LE(diff.average_abs_channel_diff, 3.0);
			EXPECT_LE(diff.max_abs_channel_diff, 50u);
		}
	}

	checkPostInvariants();
}
