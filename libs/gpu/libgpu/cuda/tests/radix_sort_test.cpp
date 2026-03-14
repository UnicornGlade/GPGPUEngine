#include <gtest/gtest.h>

#include <algorithm>

#include <libbase/stats.h>
#include <libbase/runtime_assert.h>
#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/context.h>
#include <libgpu/device.h>
#include <libgpu/shared_device_buffer.h>

#include "kernels/kernels.h"

namespace {

std::vector<gpu::Device> enumCUDADevices()
{
	std::vector<gpu::Device> devices = gpu::enumDevices(false, true, true);

	std::vector<gpu::Device> cuda_devices;
	for (auto &device : devices) {
		if (device.supports_cuda) {
			cuda_devices.push_back(device);
		}
	}
	std::cout << "CUDA supported devices: " << cuda_devices.size() << " out of " << devices.size() << std::endl;

	return cuda_devices;
}

gpu::Context activateCUDAContext(gpu::Device &device)
{
	rassert(device.supports_cuda, 2026031500375600005);
	device.supports_opencl = false;
	device.supports_vulkan = false;

	rassert(device.printInfo(), 2026031500375600006);

	gpu::Context gpu_context;
	gpu_context.init(device.device_id_cuda);
	gpu_context.activate();
	return gpu_context;
}

} // namespace

TEST(cuda, radixSort)
{
#ifndef CUDA_SUPPORT
	GTEST_SKIP() << "CUDA support is disabled in this build";
#else
	static constexpr unsigned int BLOCK_THREADS = 256;
	static constexpr unsigned int THREAD_ELEMS = 32;
	static constexpr unsigned int BLOCK_ELEMS = BLOCK_THREADS * THREAD_ELEMS;
	static constexpr unsigned int BITS_AT_A_TIME = 4;
	static constexpr unsigned int BINS_CNT = 1u << BITS_AT_A_TIME;
	static constexpr unsigned int BINS_IN_NUM = sizeof(unsigned int) << 1;

	std::vector<gpu::Device> devices = enumCUDADevices();
	if (devices.empty()) {
		GTEST_SKIP() << "No CUDA devices available";
	}

	const unsigned int n = 1 * 1000 * 1000;
	const unsigned int max_value = std::numeric_limits<int>::max();

	FastRandom r(239);
	std::vector<unsigned int> input(n, 0);
	for (size_t i = 0; i < n; ++i) {
		input[i] = r.next(0, max_value);
	}

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing CUDA device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateCUDAContext(devices[k]);

		std::vector<unsigned int> expected = input;
		timer cpu_timer;
		std::sort(expected.begin(), expected.end());
		std::cout << "CPU std::sort finished in " << cpu_timer.elapsed() << " sec" << std::endl;
		double cpu_memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
		std::cout << "CPU std::sort effective RAM bandwidth: "
				  << cpu_memory_size_gb / cpu_timer.elapsed() << " GB/s ("
				  << n / 1000.0 / 1000.0 / cpu_timer.elapsed() << " uint millions/s)" << std::endl;

		const unsigned int blocks_cnt = (n + BLOCK_ELEMS - 1u) / BLOCK_ELEMS;

		gpu::gpu_mem_32u input_gpu(n);
		gpu::gpu_mem_32u tmp_gpu(n);
		gpu::gpu_mem_32u output_gpu(n);
		gpu::gpu_mem_32u block_data_gpu(blocks_cnt << BITS_AT_A_TIME);
		gpu::gpu_mem_32u bin_counter_gpu(BINS_CNT);
		gpu::gpu_mem_32u bin_base_gpu(BINS_CNT);

		input_gpu.writeN(input.data(), n);

		std::vector<double> gpu_times;
		for (int iter = 0; iter < 10; ++iter) {
			timer gpu_timer;
			for (unsigned int pass = 0; pass < BINS_IN_NUM; ++pass) {
				const unsigned int shift = pass << 2;
				const gpu::gpu_mem_32u &in = (pass == 0) ? input_gpu : ((pass & 1u) ? tmp_gpu : output_gpu);
				gpu::gpu_mem_32u &out = (pass == 0) ? tmp_gpu : ((pass & 1u) ? output_gpu : tmp_gpu);

				cuda::radix_sort_01_local_counting(in, block_data_gpu, n, shift, blocks_cnt);
				cuda::radix_sort_02_global_prefixes_scan_sum_reduction(block_data_gpu, block_data_gpu, bin_counter_gpu, blocks_cnt);
				cuda::radix_sort_03_global_prefixes_scan_accumulation(bin_counter_gpu, bin_base_gpu);
				cuda::radix_sort_04_scatter(in, block_data_gpu, bin_base_gpu, out, n, shift, blocks_cnt);
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
			rassert(expected[i] == got[i], 2026031500375600007, expected[i], got[i], i);
		}

		std::vector<unsigned int> input_after = input_gpu.readVector();
		for (size_t i = 0; i < n; ++i) {
			rassert(input_after[i] == input[i], 2026031500375600008, input_after[i], input[i], i);
		}
	}
#endif
}
