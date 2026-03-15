#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <libbase/runtime_assert.h>
#include <libbase/fast_random.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>

#include "test_utils.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"

namespace {

struct VulkanFillCopyParams {
	std::uint32_t n;
};

struct VulkanLocalCountingParams {
	std::uint32_t shift;
	std::uint32_t n;
};

struct VulkanReductionParams {
	std::uint32_t buf1_sz;
	std::uint32_t buf2_sz;
};

struct VulkanAccumulationParams {
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

		avk2::KernelSource copy_kernel(avk2::getFillBufferWithZerosKernel());
		avk2::KernelSource local_counting(avk2::getRadixSort01LocalCountingKernel());
		avk2::KernelSource reduction(avk2::getRadixSort02GlobalPrefixesScanSumReductionKernel());
		avk2::KernelSource accumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulationKernel());
		avk2::KernelSource scatter(avk2::getRadixSort04ScatterKernel());

		gpu::gpu_mem_32u input_gpu(n);
		gpu::gpu_mem_32u tmp_gpu(n);
		gpu::gpu_mem_32u output_gpu(n);
		const unsigned int block_hist_size = div_ceil(n, static_cast<unsigned int>(BLOCK_THREADS)) * static_cast<unsigned int>(BINS_CNT);
		gpu::gpu_mem_32u block_hist_gpu(block_hist_size);
		std::vector<gpu::gpu_mem_32u> reduction_buffers;
		std::vector<unsigned int> reduction_sizes;

		unsigned int current_size = block_hist_size;
		while (current_size > BINS_CNT) {
			if ((current_size & BINS_CNT) == BINS_CNT) {
				current_size += BINS_CNT;
			}
			current_size >>= 1u;
			reduction_sizes.push_back(current_size);
			reduction_buffers.emplace_back(current_size);
		}
		rassert(!reduction_buffers.empty(), 2026031502014000006, block_hist_size);

		input_gpu.writeN(input.data(), n);

		std::vector<double> gpu_times;
		for (int iter = 0; iter < 10; ++iter) {
			timer gpu_timer;
			for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
				const unsigned int shift = pass << 2;
				const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : tmp_gpu;

				local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu);
				reduction.exec(VulkanReductionParams{block_hist_size, reduction_sizes.front()}, gpu::WorkSize(BLOCK_THREADS, reduction_sizes.front()), block_hist_gpu, reduction_buffers.front());
				for (size_t level = 1; level < reduction_buffers.size(); ++level) {
					reduction.exec(VulkanReductionParams{reduction_sizes[level - 1], reduction_sizes[level]},
								   gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]),
								   reduction_buffers[level - 1],
								   reduction_buffers[level]);
				}
				for (int level = static_cast<int>(reduction_buffers.size()) - 2; level >= 0; --level) {
					accumulation.exec(VulkanAccumulationParams{reduction_sizes[level]},
									  gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]),
									  reduction_buffers[level],
									  reduction_buffers[level + 1]);
				}
				accumulation.exec(VulkanAccumulationParams{block_hist_size},
								  gpu::WorkSize(BLOCK_THREADS, block_hist_size),
								  block_hist_gpu,
								  reduction_buffers.front());

				scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu, output_gpu, reduction_buffers.back());
				if (pass + 1u < (sizeof(unsigned int) << 1)) {
					copy_kernel.exec(VulkanFillCopyParams{n}, gpu::WorkSize(BLOCK_THREADS, n), tmp_gpu, output_gpu);
				}
			}
			gpu_times.push_back(gpu_timer.elapsed());
		}
		context.vk()->resetAsyncComputeStats();
		for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
			const unsigned int shift = pass << 2;
			const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : tmp_gpu;

			local_counting.exec(VulkanLocalCountingParams{shift, n}, gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu);
			reduction.exec(VulkanReductionParams{block_hist_size, reduction_sizes.front()}, gpu::WorkSize(BLOCK_THREADS, reduction_sizes.front()), block_hist_gpu, reduction_buffers.front());
			for (size_t level = 1; level < reduction_buffers.size(); ++level) {
				reduction.exec(VulkanReductionParams{reduction_sizes[level - 1], reduction_sizes[level]},
							   gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]),
							   reduction_buffers[level - 1],
							   reduction_buffers[level]);
			}
			for (int level = static_cast<int>(reduction_buffers.size()) - 2; level >= 0; --level) {
				accumulation.exec(VulkanAccumulationParams{reduction_sizes[level]},
								  gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]),
								  reduction_buffers[level],
								  reduction_buffers[level + 1]);
			}
			accumulation.exec(VulkanAccumulationParams{block_hist_size},
							  gpu::WorkSize(BLOCK_THREADS, block_hist_size),
							  block_hist_gpu,
							  reduction_buffers.front());

			scatter.exec(VulkanScatterParams{n, shift}, gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu, output_gpu, reduction_buffers.back());
			if (pass + 1u < (sizeof(unsigned int) << 1)) {
				copy_kernel.exec(VulkanFillCopyParams{n}, gpu::WorkSize(BLOCK_THREADS, n), tmp_gpu, output_gpu);
			}
		}
		context.vk()->logAsyncComputeStats("radixSort");

		std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(gpu_times) << std::endl;
		double gpu_memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
		std::cout << "GPU radix-sort median effective VRAM bandwidth: "
				  << gpu_memory_size_gb / stats::median(gpu_times) << " GB/s ("
				  << n / 1000.0 / 1000.0 / stats::median(gpu_times) << " uint millions/s)" << std::endl;

		std::vector<unsigned int> got = output_gpu.readVector();
		for (size_t i = 0; i < n; ++i) {
			rassert(expected[i] == got[i], 2026031502014000007, expected[i], got[i], i);
		}

		std::vector<unsigned int> input_after = input_gpu.readVector();
		for (size_t i = 0; i < n; ++i) {
			rassert(input_after[i] == input[i], 2026031502014000008, input_after[i], input[i], i);
		}
	}

	checkPostInvariants();
}
