#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <libbase/runtime_assert.h>
#include <libbase/fast_random.h>
#include <libbase/nvtx_markers.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>

#include "test_utils.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"

namespace {
struct VulkanLocalCountingParams {
	std::uint32_t shift;
	std::uint32_t n;
};

struct VulkanReductionParams {
	std::uint32_t n;
};

struct VulkanAccumulationParams {
	std::uint32_t n;
};

struct VulkanScatterParams {
	std::uint32_t n;
	std::uint32_t shift;
};

std::vector<unsigned int> makeInput(unsigned int n, unsigned int max_value)
{
	FastRandom random(239);
	std::vector<unsigned int> input(n, 0);
	for (size_t i = 0; i < n; ++i) {
		input[i] = random.next(0, max_value);
	}
	if (input.size() > 1) {
		input.back() = input.front();
	}
	return input;
}

} // namespace

TEST(vulkan, radixSort)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	const unsigned int n = 1u << 20;
	const unsigned int max_value = std::numeric_limits<int>::max();
	const std::vector<unsigned int> input = makeInput(n, max_value);
	std::vector<unsigned int> expected = input;

	timer cpu_timer;
	std::sort(expected.begin(), expected.end());
	std::cout << "CPU std::sort finished in " << cpu_timer.elapsed() << " sec" << std::endl;
	double cpu_memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
	std::cout << "CPU std::sort effective RAM bandwidth: "
			  << cpu_memory_size_gb / cpu_timer.elapsed() << " GB/s ("
			  << n / 1000.0 / 1000.0 / cpu_timer.elapsed() << " uint millions/s)" << std::endl;

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.setMemoryGuardsChecksAfterKernels(false);

		avk2::KernelSource local_counting(avk2::getRadixSort01LocalCountingKernel());
		avk2::KernelSource reduction(avk2::getRadixSort02GlobalPrefixesScanSumReductionKernel());
		avk2::KernelSource accumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulationKernel());
		avk2::KernelSource scatter(avk2::getRadixSort04ScatterKernel());

		gpu::gpu_mem_32u input_gpu(n);
		gpu::gpu_mem_32u tmp_gpu(n);
		gpu::gpu_mem_32u output_gpu(n);
		const unsigned int blocks_cnt = div_ceil(n, static_cast<unsigned int>(BLOCK_THREADS));
		const unsigned int block_hist_size = blocks_cnt * static_cast<unsigned int>(BINS_CNT);
		gpu::gpu_mem_32u block_hist_gpu(block_hist_size);
		gpu::gpu_mem_32u block_offsets_gpu(block_hist_size);
		gpu::gpu_mem_32u bin_counter_gpu(BINS_CNT);
		gpu::gpu_mem_32u bin_base_gpu(BINS_CNT);

		{
			profiling::ScopedRange scope("vk radix upload input", 0xFF1ABC9C);
			input_gpu.writeN(input.data(), n);
		}

		std::vector<double> gpu_times;
		for (int iter = 0; iter < 10; ++iter) {
			timer gpu_timer;
			for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
				profiling::ScopedRange pass_scope("vk radix pass", 0xFF4472C4);
				const unsigned int shift = pass << 2;
				const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : ((pass & 1u) ? tmp_gpu : output_gpu);
				gpu::gpu_mem_32u& out = (pass & 1u) ? output_gpu : tmp_gpu;

				{
					profiling::ScopedRange phase_scope("vk radix local_counting", 0xFF2E8B57);
					local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu);
				}
				{
					profiling::ScopedRange phase_scope("vk radix reduction", 0xFFE67E22);
					reduction.exec(VulkanReductionParams{blocks_cnt},
								   gpu::WorkSize1DTo2D(BLOCK_THREADS, BLOCK_THREADS * BINS_CNT),
								   block_hist_gpu, block_offsets_gpu, bin_counter_gpu);
				}

				{
					profiling::ScopedRange phase_scope("vk radix accumulation", 0xFF8E44AD);
					accumulation.exec(VulkanAccumulationParams{BINS_CNT},
									  gpu::WorkSize1DTo2D(BLOCK_THREADS, BLOCK_THREADS),
									  bin_counter_gpu, bin_base_gpu);
				}

				{
					profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
					scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_offsets_gpu, out, bin_base_gpu);
				}
			}
			gpu_times.push_back(gpu_timer.elapsed());
		}
		context.vk()->resetAsyncComputeStats();
		for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
			profiling::ScopedRange pass_scope("vk radix pass", 0xFF4472C4);
			const unsigned int shift = pass << 2;
			const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : ((pass & 1u) ? tmp_gpu : output_gpu);
			gpu::gpu_mem_32u& out = (pass & 1u) ? output_gpu : tmp_gpu;

			{
				profiling::ScopedRange phase_scope("vk radix local_counting", 0xFF2E8B57);
				local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu);
			}
			{
				profiling::ScopedRange phase_scope("vk radix reduction", 0xFFE67E22);
				reduction.exec(VulkanReductionParams{blocks_cnt},
							   gpu::WorkSize1DTo2D(BLOCK_THREADS, BLOCK_THREADS * BINS_CNT),
							   block_hist_gpu, block_offsets_gpu, bin_counter_gpu);
			}

			{
				profiling::ScopedRange phase_scope("vk radix accumulation", 0xFF8E44AD);
				accumulation.exec(VulkanAccumulationParams{BINS_CNT},
								  gpu::WorkSize1DTo2D(BLOCK_THREADS, BLOCK_THREADS),
								  bin_counter_gpu, bin_base_gpu);
			}

			{
				profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
				scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_offsets_gpu, out, bin_base_gpu);
			}
		}
		context.vk()->logAsyncComputeStats("radixSort");

		std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(gpu_times) << std::endl;
		double gpu_memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
		std::cout << "GPU radix-sort median effective VRAM bandwidth: "
				  << gpu_memory_size_gb / stats::median(gpu_times) << " GB/s ("
				  << n / 1000.0 / 1000.0 / stats::median(gpu_times) << " uint millions/s)" << std::endl;

		std::vector<unsigned int> got;
		{
			profiling::ScopedRange scope("vk radix read output", 0xFFF1C40F);
			got = output_gpu.readVector();
		}
		for (size_t i = 0; i < n; ++i) {
			rassert(expected[i] == got[i], 2026031502014000007, expected[i], got[i], i);
		}

		std::vector<unsigned int> input_after;
		{
			profiling::ScopedRange scope("vk radix read input_after", 0xFFF1C40F);
			input_after = input_gpu.readVector();
		}
		for (size_t i = 0; i < n; ++i) {
			rassert(input_after[i] == input[i], 2026031502014000008, input_after[i], input[i], i);
		}
	}

	checkPostInvariants();
}
