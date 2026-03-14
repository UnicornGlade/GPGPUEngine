#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

#include <libbase/runtime_assert.h>
#include <libbase/fast_random.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

namespace {

std::vector<gpu::Device> enumOpenCLDevices()
{
	gpu::Device::enable_cuda = false;
	std::vector<gpu::Device> devices = gpu::enumDevices(false, false, false);

	std::vector<gpu::Device> ocl_devices;
	for (auto &device : devices) {
		if (device.supports_opencl) {
			ocl_devices.push_back(device);
		}
	}
	std::cout << "OpenCL supported devices: " << ocl_devices.size() << " out of " << devices.size() << std::endl;

	return ocl_devices;
}

gpu::Context activateOpenCLContext(gpu::Device &device)
{
	rassert(device.supports_opencl, 2026031502014000001);
	device.supports_cuda = false;
	device.supports_vulkan = false;

	rassert(device.printInfo(), 2026031502014000002);

	gpu::Context gpu_context;
	gpu_context.init(device.device_id_opencl);
	gpu_context.activate();
	return gpu_context;
}

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

TEST(opencl, radixSort)
{
	std::vector<gpu::Device> devices = enumOpenCLDevices();
	if (devices.empty()) {
		GTEST_SKIP() << "No OpenCL devices available";
	}

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
		std::cout << "Testing OpenCL device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateOpenCLContext(devices[k]);

		ocl::KernelSource copy_kernel(ocl::getFillBufferWithZerosKernel());
		ocl::KernelSource local_counting(ocl::getRadixSort01LocalCountingKernel());
		ocl::KernelSource reduction(ocl::getRadixSort02GlobalPrefixesScanSumReductionKernel());
		ocl::KernelSource accumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulationKernel());
		ocl::KernelSource scatter(ocl::getRadixSort04ScatterKernel());

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
		rassert(!reduction_buffers.empty(), 2026031502014000003, block_hist_size);

		input_gpu.writeN(input.data(), n);

		std::vector<double> gpu_times;
		for (int iter = 0; iter < 10; ++iter) {
			timer gpu_timer;
			for (unsigned int pass = 0; pass < (sizeof(unsigned int) << 1); ++pass) {
				const unsigned int shift = pass << 2;
				const gpu::gpu_mem_32u& in = (pass == 0u) ? input_gpu : tmp_gpu;
				local_counting.exec(gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu, shift, n);

				reduction.exec(gpu::WorkSize(BLOCK_THREADS, reduction_sizes.front()), block_hist_gpu, reduction_buffers.front(), block_hist_size, reduction_sizes.front());
				for (size_t level = 1; level < reduction_buffers.size(); ++level) {
					reduction.exec(gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]), reduction_buffers[level - 1], reduction_buffers[level], reduction_sizes[level - 1], reduction_sizes[level]);
				}
				for (int level = static_cast<int>(reduction_buffers.size()) - 2; level >= 0; --level) {
					accumulation.exec(gpu::WorkSize(BLOCK_THREADS, reduction_sizes[level]), reduction_buffers[level], reduction_buffers[level + 1], reduction_sizes[level]);
				}
				accumulation.exec(gpu::WorkSize(BLOCK_THREADS, block_hist_size), block_hist_gpu, reduction_buffers.front(), block_hist_size);

				scatter.exec(gpu::WorkSize(BLOCK_THREADS, n), in, block_hist_gpu, output_gpu, reduction_buffers.back(), n, shift);
				if (pass + 1u < (sizeof(unsigned int) << 1)) {
					copy_kernel.exec(gpu::WorkSize(BLOCK_THREADS, n), tmp_gpu, output_gpu, n);
				}
			}
			gpu_times.push_back(gpu_timer.elapsed());
		}

		std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(gpu_times) << std::endl;
		double gpu_memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
		std::cout << "GPU radix-sort median effective VRAM bandwidth: "
				  << gpu_memory_size_gb / stats::median(gpu_times) << " GB/s ("
				  << n / 1000.0 / 1000.0 / stats::median(gpu_times) << " uint millions/s)" << std::endl;

		std::vector<unsigned int> got = output_gpu.readVector();
		for (size_t i = 0; i < n; ++i) {
			rassert(expected[i] == got[i], 2026031502014000004, expected[i], got[i], i);
		}

		std::vector<unsigned int> input_after = input_gpu.readVector();
		for (size_t i = 0; i < n; ++i) {
			rassert(input_after[i] == input[i], 2026031502014000005, input_after[i], input[i], i);
		}
	}
}
