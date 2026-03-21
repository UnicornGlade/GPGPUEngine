#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <libbase/nvtx_markers.h>
#include <libbase/runtime_assert.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/vulkan/device.h>

#include "kernels/kernels.h"
#include "test_utils.h"

namespace {

constexpr uint32_t kDescriptorDim = 128u;
constexpr uint32_t kDefaultDescriptorCount = 40000u;
constexpr uint32_t kDefaultBenchmarkIterations = 3u;
constexpr uint32_t kBruteforceLocalGroupSize = 64u;
constexpr uint32_t kFp16PackedGroupSize = 32u;
constexpr uint32_t kGemmLikeGroupSize = 16u;
constexpr uint32_t kGemmLikeVec4GroupSize = 32u;

struct MatchResult {
	std::vector<uint32_t> indices;
	std::vector<float> scores;
};

struct CpuBenchmarkResult {
	MatchResult matches;
	double seconds = 0.0;
};

struct GpuBenchmarkResult {
	MatchResult matches;
	double upload_seconds = 0.0;
	std::vector<double> compute_seconds;
	double readback_seconds = 0.0;
};

struct MatchPushConstants {
	uint32_t n_queries;
	uint32_t n_trains;
};

uint32_t getenvU32(const char *name, uint32_t default_value)
{
	const char *value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}

	char *end = nullptr;
	const unsigned long parsed = std::strtoul(value, &end, 10);
	rassert(end != value && *end == '\0', 2026032123250100001, name, value);
	rassert(parsed > 0ul, 2026032123250100002, name, value);
	rassert(parsed <= std::numeric_limits<uint32_t>::max(), 2026032123250100003, name, value);
	return static_cast<uint32_t>(parsed);
}

uint32_t selectCpuThreadCount()
{
	const uint32_t default_threads = std::max(1u, std::thread::hardware_concurrency());
	return getenvU32("GPGPU_VULKAN_SIFT_MATCH_CPU_THREADS", default_threads);
}

double computeGflops(uint32_t n_queries, uint32_t n_trains, double seconds)
{
	rassert(seconds > 0.0, 2026032123250100004, seconds);
	const double total_flops = 2.0 * static_cast<double>(n_queries) * static_cast<double>(n_trains) * static_cast<double>(kDescriptorDim);
	return total_flops / seconds / 1e9;
}

float computeDot(const std::vector<float> &queries,
				 const std::vector<float> &trains,
				 uint32_t query_index,
				 uint32_t train_index)
{
	const size_t query_offset = static_cast<size_t>(query_index) * kDescriptorDim;
	const size_t train_offset = static_cast<size_t>(train_index) * kDescriptorDim;
	float score = 0.0f;
	for (uint32_t component = 0; component < kDescriptorDim; ++component) {
		score += queries[query_offset + component] * trains[train_offset + component];
	}
	return score;
}

uint16_t floatToHalfBits(float value)
{
	union {
		float f;
		uint32_t u;
	} bits = {value};

	const uint32_t sign = (bits.u >> 16u) & 0x8000u;
	int32_t exponent = static_cast<int32_t>((bits.u >> 23u) & 0xFFu) - 127 + 15;
	uint32_t mantissa = bits.u & 0x7FFFFFu;

	if (exponent <= 0) {
		if (exponent < -10) {
			return static_cast<uint16_t>(sign);
		}
		mantissa |= 0x800000u;
		const uint32_t shifted = mantissa >> static_cast<uint32_t>(1 - exponent + 13);
		const uint32_t rounded = shifted + ((mantissa >> static_cast<uint32_t>(1 - exponent + 12)) & 1u);
		return static_cast<uint16_t>(sign | rounded);
	}
	if (exponent >= 31) {
		return static_cast<uint16_t>(sign | 0x7C00u);
	}

	const uint32_t rounded_mantissa = mantissa + 0x1000u;
	if (rounded_mantissa & 0x800000u) {
		mantissa = 0u;
		++exponent;
		if (exponent >= 31) {
			return static_cast<uint16_t>(sign | 0x7C00u);
		}
	} else {
		mantissa = rounded_mantissa;
	}

	return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10u) | (mantissa >> 13u));
}

float halfBitsToFloat(uint16_t value)
{
	const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16u;
	const uint32_t exponent = (value >> 10u) & 0x1Fu;
	const uint32_t mantissa = value & 0x03FFu;

	uint32_t float_bits = 0u;
	if (exponent == 0u) {
		if (mantissa == 0u) {
			float_bits = sign;
		} else {
			uint32_t mantissa_norm = mantissa;
			int32_t exponent_norm = -14;
			while ((mantissa_norm & 0x0400u) == 0u) {
				mantissa_norm <<= 1u;
				--exponent_norm;
			}
			mantissa_norm &= 0x03FFu;
			float_bits = sign
					   | (static_cast<uint32_t>(exponent_norm + 127) << 23u)
					   | (mantissa_norm << 13u);
		}
	} else if (exponent == 0x1Fu) {
		float_bits = sign | 0x7F800000u | (mantissa << 13u);
	} else {
		float_bits = sign
				   | ((exponent + 112u) << 23u)
				   | (mantissa << 13u);
	}

	union {
		uint32_t u;
		float f;
	} bits = {float_bits};
	return bits.f;
}

std::vector<uint32_t> packDescriptorsToFp16(const std::vector<float> &descriptors)
{
	rassert(descriptors.size() % 2u == 0u, 2026032123594300001, descriptors.size());
	std::vector<uint32_t> packed(descriptors.size() / 2u, 0u);
	for (size_t i = 0; i < packed.size(); ++i) {
		const uint16_t lo = floatToHalfBits(descriptors[2u * i + 0u]);
		const uint16_t hi = floatToHalfBits(descriptors[2u * i + 1u]);
		packed[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16u);
	}
	return packed;
}

std::vector<float> unpackDescriptorsFromFp16(const std::vector<uint32_t> &packed)
{
	std::vector<float> descriptors(packed.size() * 2u, 0.0f);
	for (size_t i = 0; i < packed.size(); ++i) {
		const uint16_t lo = static_cast<uint16_t>(packed[i] & 0xFFFFu);
		const uint16_t hi = static_cast<uint16_t>(packed[i] >> 16u);
		descriptors[2u * i + 0u] = halfBitsToFloat(lo);
		descriptors[2u * i + 1u] = halfBitsToFloat(hi);
	}
	return descriptors;
}

std::vector<float> makeRandomRootSiftDescriptors(uint32_t descriptor_count, uint32_t seed)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	std::vector<float> descriptors(static_cast<size_t>(descriptor_count) * kDescriptorDim, 0.0f);
	for (uint32_t descriptor_index = 0; descriptor_index < descriptor_count; ++descriptor_index) {
		const size_t offset = static_cast<size_t>(descriptor_index) * kDescriptorDim;
		float sum = 0.0f;
		for (uint32_t component = 0; component < kDescriptorDim; ++component) {
			const float value = dist(rng);
			descriptors[offset + component] = value;
			sum += value;
		}

		const float inv_l1 = 1.0f / std::max(sum, 1e-12f);
		for (uint32_t component = 0; component < kDescriptorDim; ++component) {
			descriptors[offset + component] = std::sqrt(descriptors[offset + component] * inv_l1);
		}
	}

	return descriptors;
}

CpuBenchmarkResult bruteForceCpuReference(const std::vector<float> &queries,
										  const std::vector<float> &trains,
										  uint32_t n_queries,
										  uint32_t n_trains,
										  uint32_t thread_count)
{
	rassert(queries.size() == static_cast<size_t>(n_queries) * kDescriptorDim, 2026032123250100005, queries.size(), n_queries);
	rassert(trains.size() == static_cast<size_t>(n_trains) * kDescriptorDim, 2026032123250100006, trains.size(), n_trains);
	rassert(thread_count > 0u, 2026032123250100007, thread_count);

	CpuBenchmarkResult result;
	result.matches.indices.resize(n_queries, 0u);
	result.matches.scores.resize(n_queries, -std::numeric_limits<float>::infinity());

	timer cpu_timer;
	std::vector<std::thread> workers;
	workers.reserve(thread_count);
	for (uint32_t thread_index = 0; thread_index < thread_count; ++thread_index) {
		workers.emplace_back([&, thread_index]() {
			const uint32_t query_begin = static_cast<uint32_t>((static_cast<uint64_t>(n_queries) * thread_index) / thread_count);
			const uint32_t query_end = static_cast<uint32_t>((static_cast<uint64_t>(n_queries) * (thread_index + 1u)) / thread_count);
			for (uint32_t query_index = query_begin; query_index < query_end; ++query_index) {
				float best_score = -std::numeric_limits<float>::infinity();
				uint32_t best_index = 0u;
				for (uint32_t train_index = 0; train_index < n_trains; ++train_index) {
					const float score = computeDot(queries, trains, query_index, train_index);
					if (score > best_score) {
						best_score = score;
						best_index = train_index;
					}
				}
				result.matches.indices[query_index] = best_index;
				result.matches.scores[query_index] = best_score;
			}
		});
	}
	for (std::thread &worker : workers) {
		worker.join();
	}
	result.seconds = cpu_timer.elapsed();
	return result;
}

GpuBenchmarkResult runGpuMatcher(avk2::KernelSource &kernel,
								 gpu::Context &context,
								 uint32_t group_size,
								 const std::vector<float> &queries,
								 const std::vector<float> &trains,
								 uint32_t n_queries,
								 uint32_t n_trains,
								 uint32_t benchmark_iterations)
{
	GpuBenchmarkResult result;

	gpu::gpu_mem_32f queries_gpu(n_queries * kDescriptorDim);
	gpu::gpu_mem_32f trains_gpu(n_trains * kDescriptorDim);
	gpu::gpu_mem_32u best_indices_gpu(n_queries);
	gpu::gpu_mem_32f best_scores_gpu(n_queries);

	{
		timer upload_timer;
		profiling::ScopedRange scope("sift match upload", 0xFF236FA1);
		queries_gpu.writeN(queries.data(), queries.size());
		trains_gpu.writeN(trains.data(), trains.size());
		context.vk()->waitForAllInflightComputeLaunches();
		result.upload_seconds = upload_timer.elapsed();
	}

	MatchPushConstants params{n_queries, n_trains};
	const gpu::WorkSize work_size = gpu::WorkSize1DTo2D(group_size, n_queries);

	{
		profiling::ScopedRange scope("sift match gpu warmup", 0xFF7E57C2);
		kernel.exec(params, work_size, queries_gpu, trains_gpu, best_indices_gpu, best_scores_gpu);
		context.vk()->waitForAllInflightComputeLaunches();
	}

	for (uint32_t iter = 0; iter < benchmark_iterations; ++iter) {
		timer compute_timer;
		{
			profiling::ScopedRange scope("sift match gpu compute", 0xFF2E8B57);
			kernel.exec(params, work_size, queries_gpu, trains_gpu, best_indices_gpu, best_scores_gpu);
			context.vk()->waitForAllInflightComputeLaunches();
		}
		result.compute_seconds.push_back(compute_timer.elapsed());
	}

	{
		timer readback_timer;
		profiling::ScopedRange scope("sift match readback", 0xFFB9770E);
		result.matches.indices = best_indices_gpu.readVector(n_queries);
		result.matches.scores = best_scores_gpu.readVector(n_queries);
		result.readback_seconds = readback_timer.elapsed();
	}

	return result;
}

GpuBenchmarkResult runGpuMatcherFp16Packed(avk2::KernelSource &kernel,
										   gpu::Context &context,
										   uint32_t group_size,
										   const std::vector<uint32_t> &queries_packed,
										   const std::vector<uint32_t> &trains_packed,
										   uint32_t n_queries,
										   uint32_t n_trains,
										   uint32_t benchmark_iterations)
{
	GpuBenchmarkResult result;

	gpu::gpu_mem_32u queries_gpu(queries_packed.size());
	gpu::gpu_mem_32u trains_gpu(trains_packed.size());
	gpu::gpu_mem_32u best_indices_gpu(n_queries);
	gpu::gpu_mem_32f best_scores_gpu(n_queries);

	{
		timer upload_timer;
		profiling::ScopedRange scope("sift match fp16 upload", 0xFF236FA1);
		queries_gpu.writeN(queries_packed.data(), queries_packed.size());
		trains_gpu.writeN(trains_packed.data(), trains_packed.size());
		context.vk()->waitForAllInflightComputeLaunches();
		result.upload_seconds = upload_timer.elapsed();
	}

	MatchPushConstants params{n_queries, n_trains};
	const gpu::WorkSize work_size = gpu::WorkSize1DTo2D(group_size, n_queries);

	{
		profiling::ScopedRange scope("sift match fp16 warmup", 0xFF7E57C2);
		kernel.exec(params, work_size, queries_gpu, trains_gpu, best_indices_gpu, best_scores_gpu);
		context.vk()->waitForAllInflightComputeLaunches();
	}

	for (uint32_t iter = 0; iter < benchmark_iterations; ++iter) {
		timer compute_timer;
		{
			profiling::ScopedRange scope("sift match fp16 compute", 0xFF2E8B57);
			kernel.exec(params, work_size, queries_gpu, trains_gpu, best_indices_gpu, best_scores_gpu);
			context.vk()->waitForAllInflightComputeLaunches();
		}
		result.compute_seconds.push_back(compute_timer.elapsed());
	}

	{
		timer readback_timer;
		profiling::ScopedRange scope("sift match fp16 readback", 0xFFB9770E);
		result.matches.indices = best_indices_gpu.readVector(n_queries);
		result.matches.scores = best_scores_gpu.readVector(n_queries);
		result.readback_seconds = readback_timer.elapsed();
	}

	return result;
}

void expectCloseToCpuReference(const std::string &label,
							   const MatchResult &cpu_reference,
							   const MatchResult &gpu_result,
							   const std::vector<float> &queries,
							   const std::vector<float> &trains,
							   uint32_t n_trains)
{
	rassert(cpu_reference.indices.size() == gpu_result.indices.size(), 2026032123250100008, cpu_reference.indices.size(), gpu_result.indices.size());
	rassert(cpu_reference.scores.size() == gpu_result.scores.size(), 2026032123250100009, cpu_reference.scores.size(), gpu_result.scores.size());

	const float kAllowedScoreGap = 1e-4f;
	const float kAllowedReadbackError = 5e-4f;
	size_t mismatches = 0;
	float max_score_gap = 0.0f;
	float max_readback_error = 0.0f;
	std::vector<std::string> mismatch_examples;

	for (size_t query_index = 0; query_index < cpu_reference.indices.size(); ++query_index) {
		rassert(gpu_result.indices[query_index] < n_trains, 2026032123250100010, query_index, gpu_result.indices[query_index], n_trains);
		const float cpu_best = cpu_reference.scores[query_index];
		const float gpu_reported = gpu_result.scores[query_index];
		const float gpu_actual = computeDot(queries, trains, static_cast<uint32_t>(query_index), gpu_result.indices[query_index]);
		const float score_gap = cpu_best - gpu_actual;
		const float readback_error = std::abs(gpu_reported - gpu_actual);
		max_score_gap = std::max(max_score_gap, score_gap);
		max_readback_error = std::max(max_readback_error, readback_error);

		if (score_gap > kAllowedScoreGap || readback_error > kAllowedReadbackError) {
			++mismatches;
			if (mismatch_examples.size() < 5u) {
				std::ostringstream stream;
				stream << "q=" << query_index
					   << " cpu_idx=" << cpu_reference.indices[query_index]
					   << " gpu_idx=" << gpu_result.indices[query_index]
					   << " cpu_score=" << std::setprecision(8) << cpu_best
					   << " gpu_actual=" << gpu_actual
					   << " gpu_reported=" << gpu_reported
					   << " score_gap=" << score_gap
					   << " readback_error=" << readback_error;
				mismatch_examples.push_back(stream.str());
			}
		}
	}

	EXPECT_EQ(mismatches, 0u) << label
							  << " mismatches=" << mismatches
							  << " max_score_gap=" << max_score_gap
							  << " max_readback_error=" << max_readback_error
							  << (mismatch_examples.empty() ? "" : " examples: " + mismatch_examples.front());
}

bool hasFp32CooperativeMatrices(const gpu::Device &device)
{
	avk2::Device vk_device(device.device_id_vulkan);
	if (!vk_device.init(true)) {
		return false;
	}

	for (const auto &size : vk_device.supportedCooperativeMatrixSizes()) {
		if (std::get<0>(size) == DataType32f && std::get<1>(size) == DataType32f) {
			return true;
		}
	}
	return false;
}

void logCpuBenchmark(const CpuBenchmarkResult &result,
					 uint32_t n_queries,
					 uint32_t n_trains,
					 uint32_t thread_count)
{
	std::cout << "CPU brute-force RootSIFT matching: queries=" << n_queries
			  << " trains=" << n_trains
			  << " dim=" << kDescriptorDim
			  << " threads=" << thread_count
			  << " time=" << result.seconds * 1000.0 << " ms"
			  << " achieved=" << computeGflops(n_queries, n_trains, result.seconds) << " GFLOPS"
			  << std::endl;
}

void logGpuBenchmark(const std::string &label,
					 const GpuBenchmarkResult &result,
					 uint32_t n_queries,
					 uint32_t n_trains,
					 bool has_fp32_coop_matrices)
{
	const double median_seconds = stats::median(result.compute_seconds);
	std::cout << label
			  << ": queries=" << n_queries
			  << " trains=" << n_trains
			  << " dim=" << kDescriptorDim
			  << " cooperative_matrix_fp32=" << (has_fp32_coop_matrices ? "yes" : "no")
			  << std::endl;
	std::cout << "  upload median-equivalent: " << result.upload_seconds * 1000.0 << " ms" << std::endl;
	std::cout << "  gpu compute median:       " << median_seconds * 1000.0 << " ms ("
			  << stats::valuesStatsLine(result.compute_seconds) << ")" << std::endl;
	std::cout << "  readback:                 " << result.readback_seconds * 1000.0 << " ms" << std::endl;
	std::cout << "  achieved throughput:      " << computeGflops(n_queries, n_trains, median_seconds) << " GFLOPS" << std::endl;
}

void logTop1DifferencesVsFp32(const std::string &label,
							  const MatchResult &fp32_reference,
							  const MatchResult &candidate)
{
	rassert(fp32_reference.indices.size() == candidate.indices.size(), 2026032200110400001, fp32_reference.indices.size(), candidate.indices.size());
	size_t different_count = 0;
	for (size_t i = 0; i < fp32_reference.indices.size(); ++i) {
		if (fp32_reference.indices[i] != candidate.indices[i]) {
			++different_count;
		}
	}

	const double total = static_cast<double>(fp32_reference.indices.size());
	const double percent = total > 0.0 ? 100.0 * static_cast<double>(different_count) / total : 0.0;
	std::cout << "  top-1 differences vs fp32: " << different_count
			  << " / " << fp32_reference.indices.size()
			  << " (" << percent << "%)" << std::endl;
}

} // namespace

TEST(vulkan, siftMatchBenchmark)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	size_t gpu_device_count = 0;
	for (const gpu::Device &device : devices) {
		if (device.isGPU()) {
			++gpu_device_count;
		}
	}
	if (gpu_device_count == 0) {
		GTEST_SKIP() << "No GPU-class Vulkan devices found for SIFT matching benchmark";
	}

	const uint32_t descriptor_count = getenvU32("GPGPU_VULKAN_SIFT_MATCH_COUNT", kDefaultDescriptorCount);
	const uint32_t benchmark_iterations = getenvU32("GPGPU_VULKAN_SIFT_MATCH_ITERS", kDefaultBenchmarkIterations);
	const uint32_t cpu_thread_count = selectCpuThreadCount();

	std::vector<float> queries = makeRandomRootSiftDescriptors(descriptor_count, 239u);
	std::vector<float> trains = makeRandomRootSiftDescriptors(descriptor_count, 997u);
	const std::vector<uint32_t> queries_fp16_packed = packDescriptorsToFp16(queries);
	const std::vector<uint32_t> trains_fp16_packed = packDescriptorsToFp16(trains);
	const std::vector<float> queries_fp16 = unpackDescriptorsFromFp16(queries_fp16_packed);
	const std::vector<float> trains_fp16 = unpackDescriptorsFromFp16(trains_fp16_packed);

	CpuBenchmarkResult cpu_reference;
	{
		profiling::ScopedRange scope("sift match cpu reference", 0xFF5B8C5A);
		cpu_reference = bruteForceCpuReference(queries, trains, descriptor_count, descriptor_count, cpu_thread_count);
	}
	logCpuBenchmark(cpu_reference, descriptor_count, descriptor_count, cpu_thread_count);

	CpuBenchmarkResult cpu_reference_fp16;
	{
		profiling::ScopedRange scope("sift match cpu fp16 reference", 0xFF6C5CE7);
		cpu_reference_fp16 = bruteForceCpuReference(queries_fp16, trains_fp16, descriptor_count, descriptor_count, cpu_thread_count);
	}
	logCpuBenchmark(cpu_reference_fp16, descriptor_count, descriptor_count, cpu_thread_count);
	std::cout << "CPU fp16 quantized reference drift:" << std::endl;
	logTop1DifferencesVsFp32("cpu fp16 vs fp32", cpu_reference.matches, cpu_reference_fp16.matches);

	size_t tested_devices = 0;
	for (size_t device_index = 0; device_index < devices.size(); ++device_index) {
		if (!devices[device_index].isGPU()) {
			std::cout << "Skipping non-GPU Vulkan device: " << devices[device_index].name << std::endl;
			continue;
		}

		++tested_devices;
		std::cout << "Testing Vulkan device #" << tested_devices << ": " << devices[device_index].name << std::endl;
		gpu::Context context = activateVKContext(devices[device_index]);
		context.setMemoryGuardsChecksAfterKernels(false);

		avk2::KernelSource brute_force_kernel(avk2::getSiftMatchBruteforceLocalKernel());
		avk2::KernelSource fp16_packed_kernel(avk2::getSiftMatchGemmLikeFp16PackedKernel());
		avk2::KernelSource gemm_like_kernel(avk2::getSiftMatchGemmLikeKernel());
		avk2::KernelSource gemm_like_vec4_kernel(avk2::getSiftMatchGemmLikeVec4Kernel());
		const bool has_fp32_coop_matrices = hasFp32CooperativeMatrices(devices[device_index]);

		const GpuBenchmarkResult brute_force_result = runGpuMatcher(brute_force_kernel,
																	 context,
																	 kBruteforceLocalGroupSize,
																	 queries,
																	 trains,
																	 descriptor_count,
																	 descriptor_count,
																	 benchmark_iterations);
		expectCloseToCpuReference("vk brute-force local matcher",
								  cpu_reference.matches,
								  brute_force_result.matches,
								  queries,
								  trains,
								  descriptor_count);
		logGpuBenchmark("Vulkan brute-force local SIFT matcher",
						brute_force_result,
						descriptor_count,
						descriptor_count,
						has_fp32_coop_matrices);

		const GpuBenchmarkResult fp16_packed_result = runGpuMatcherFp16Packed(fp16_packed_kernel,
																	 context,
																	 kFp16PackedGroupSize,
																	 queries_fp16_packed,
																	 trains_fp16_packed,
																	 descriptor_count,
																	 descriptor_count,
																	 benchmark_iterations);
		expectCloseToCpuReference("vk gemm-like fp16 packed matcher",
								  cpu_reference_fp16.matches,
								  fp16_packed_result.matches,
								  queries_fp16,
								  trains_fp16,
								  descriptor_count);
		logGpuBenchmark("Vulkan GEMM-like fp16 packed SIFT matcher",
						fp16_packed_result,
						descriptor_count,
						descriptor_count,
						has_fp32_coop_matrices);
		logTop1DifferencesVsFp32("gpu fp16 vs fp32", cpu_reference.matches, fp16_packed_result.matches);

		const GpuBenchmarkResult gemm_like_result = runGpuMatcher(gemm_like_kernel,
																 context,
																 kGemmLikeGroupSize,
																 queries,
																 trains,
																 descriptor_count,
																 descriptor_count,
																 benchmark_iterations);
		expectCloseToCpuReference("vk gemm-like matcher",
								  cpu_reference.matches,
								  gemm_like_result.matches,
								  queries,
								  trains,
								  descriptor_count);
		logGpuBenchmark("Vulkan GEMM-like SIFT matcher",
						gemm_like_result,
						descriptor_count,
						descriptor_count,
						has_fp32_coop_matrices);

		const GpuBenchmarkResult gemm_like_vec4_result = runGpuMatcher(gemm_like_vec4_kernel,
																	 context,
																	 kGemmLikeVec4GroupSize,
																	 queries,
																	 trains,
																	 descriptor_count,
																	 descriptor_count,
																	 benchmark_iterations);
		expectCloseToCpuReference("vk gemm-like vec4 matcher",
								  cpu_reference.matches,
								  gemm_like_vec4_result.matches,
								  queries,
								  trains,
								  descriptor_count);
		logGpuBenchmark("Vulkan GEMM-like vec4 SIFT matcher",
						gemm_like_vec4_result,
						descriptor_count,
						descriptor_count,
						has_fp32_coop_matrices);

		checkPostInvariants();
	}
}
