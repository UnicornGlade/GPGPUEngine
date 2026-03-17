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

struct VulkanPortableReductionParams {
	std::uint32_t buf1_sz;
	std::uint32_t buf2_sz;
};

struct VulkanPortableAccumulationParams {
	std::uint32_t buf1_sz;
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

		const bool use_fast_radix = devices[k].isNvidia() || devices[k].isIntel();
		std::cout << "  radix sort path: " << (use_fast_radix ? "fast-subgroup" : "portable-fallback") << std::endl;

		avk2::KernelSource local_counting(use_fast_radix ? avk2::getRadixSort01LocalCountingKernel()
														 : avk2::getRadixSort01LocalCountingPortableKernel());
		avk2::KernelSource reduction(use_fast_radix ? avk2::getRadixSort02GlobalPrefixesScanSumReductionKernel()
													: avk2::getRadixSort02GlobalPrefixesScanSumReductionPortableKernel());
		avk2::KernelSource accumulation(use_fast_radix ? avk2::getRadixSort03GlobalPrefixesScanAccumulationKernel()
													   : avk2::getRadixSort03GlobalPrefixesScanAccumulationPortableKernel());
		avk2::KernelSource scatter(use_fast_radix ? avk2::getRadixSort04ScatterKernel()
												  : avk2::getRadixSort04ScatterPortableKernel());
		avk2::KernelSource copy_kernel(avk2::getFillBufferWithZerosKernel());

		gpu::gpu_mem_32u input_gpu(n);
		gpu::gpu_mem_32u tmp_gpu(n);
		gpu::gpu_mem_32u output_gpu(n);
		const unsigned int blocks_cnt = div_ceil(n, static_cast<unsigned int>(BLOCK_THREADS));
		const unsigned int block_hist_size = blocks_cnt * static_cast<unsigned int>(BINS_CNT);
		gpu::gpu_mem_32u block_hist_gpu(block_hist_size);
		gpu::gpu_mem_32u block_offsets_gpu(block_hist_size);
		gpu::gpu_mem_32u bin_counter_gpu(BINS_CNT);
		gpu::gpu_mem_32u bin_base_gpu(BINS_CNT);
		std::vector<gpu::gpu_mem_32u> reduction_buffers;
		std::vector<unsigned int> reduction_sizes;
		if (!use_fast_radix) {
			unsigned int current_size = block_hist_size;
			while (current_size > BINS_CNT) {
				if ((current_size & BINS_CNT) == BINS_CNT) {
					current_size += BINS_CNT;
				}
				current_size >>= 1u;
				reduction_sizes.push_back(current_size);
				reduction_buffers.emplace_back(current_size);
			}
			rassert(!reduction_buffers.empty(), 2026031704373100001, block_hist_size);
		}

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
				if (use_fast_radix) {
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
						profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
						scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_offsets_gpu, out, bin_counter_gpu);
					}
				} else {
					const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : tmp_gpu;

					{
						profiling::ScopedRange phase_scope("vk radix local_counting", 0xFF2E8B57);
						local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu);
					}
					{
						profiling::ScopedRange phase_scope("vk radix reduction root", 0xFFE67E22);
						reduction.exec(VulkanPortableReductionParams{block_hist_size, reduction_sizes.front()},
									   gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes.front()),
									   block_hist_gpu, reduction_buffers.front());
					}
					for (size_t level = 1; level < reduction_buffers.size(); ++level) {
						profiling::ScopedRange phase_scope("vk radix reduction level", 0xFFE67E22);
						reduction.exec(VulkanPortableReductionParams{reduction_sizes[level - 1], reduction_sizes[level]},
									   gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes[level]),
									   reduction_buffers[level - 1],
									   reduction_buffers[level]);
					}
					for (int level = static_cast<int>(reduction_buffers.size()) - 2; level >= 0; --level) {
						profiling::ScopedRange phase_scope("vk radix accumulation level", 0xFF8E44AD);
						accumulation.exec(VulkanPortableAccumulationParams{reduction_sizes[level]},
										  gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes[level]),
										  reduction_buffers[level],
										  reduction_buffers[level + 1]);
					}
					{
						profiling::ScopedRange phase_scope("vk radix accumulation root", 0xFF8E44AD);
						accumulation.exec(VulkanPortableAccumulationParams{block_hist_size},
										  gpu::WorkSize1DTo2D(BLOCK_THREADS, block_hist_size),
										  block_hist_gpu,
										  reduction_buffers.front());
					}
					{
						profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
						scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu, output_gpu, reduction_buffers.back());
					}
					if (pass + 1u < (sizeof(unsigned int) << 1)) {
						profiling::ScopedRange phase_scope("vk radix copy", 0xFF16A085);
						copy_kernel.exec(n, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), tmp_gpu, output_gpu);
					}
				}
			}
			gpu_times.push_back(gpu_timer.elapsed());
		}
		context.vk()->resetAsyncComputeStats();
		for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
			profiling::ScopedRange pass_scope("vk radix pass", 0xFF4472C4);
			const unsigned int shift = pass << 2;
			if (use_fast_radix) {
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
					profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
					scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_offsets_gpu, out, bin_counter_gpu);
				}
			} else {
				const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : tmp_gpu;

				{
					profiling::ScopedRange phase_scope("vk radix local_counting", 0xFF2E8B57);
					local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu);
				}
				{
					profiling::ScopedRange phase_scope("vk radix reduction root", 0xFFE67E22);
					reduction.exec(VulkanPortableReductionParams{block_hist_size, reduction_sizes.front()},
								   gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes.front()),
								   block_hist_gpu, reduction_buffers.front());
				}
				for (size_t level = 1; level < reduction_buffers.size(); ++level) {
					profiling::ScopedRange phase_scope("vk radix reduction level", 0xFFE67E22);
					reduction.exec(VulkanPortableReductionParams{reduction_sizes[level - 1], reduction_sizes[level]},
								   gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes[level]),
								   reduction_buffers[level - 1],
								   reduction_buffers[level]);
				}
				for (int level = static_cast<int>(reduction_buffers.size()) - 2; level >= 0; --level) {
					profiling::ScopedRange phase_scope("vk radix accumulation level", 0xFF8E44AD);
					accumulation.exec(VulkanPortableAccumulationParams{reduction_sizes[level]},
									  gpu::WorkSize1DTo2D(BLOCK_THREADS, reduction_sizes[level]),
									  reduction_buffers[level],
									  reduction_buffers[level + 1]);
				}
				{
					profiling::ScopedRange phase_scope("vk radix accumulation root", 0xFF8E44AD);
					accumulation.exec(VulkanPortableAccumulationParams{block_hist_size},
									  gpu::WorkSize1DTo2D(BLOCK_THREADS, block_hist_size),
									  block_hist_gpu,
									  reduction_buffers.front());
				}

				{
					profiling::ScopedRange phase_scope("vk radix scatter", 0xFFC0392B);
					scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), in, block_hist_gpu, output_gpu, reduction_buffers.back());
				}
				if (pass + 1u < (sizeof(unsigned int) << 1)) {
					profiling::ScopedRange phase_scope("vk radix copy", 0xFF16A085);
					copy_kernel.exec(n, gpu::WorkSize1DTo2D(BLOCK_THREADS, n), tmp_gpu, output_gpu);
				}
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
