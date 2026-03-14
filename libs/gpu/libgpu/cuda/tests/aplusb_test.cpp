#include <gtest/gtest.h>

#include <libbase/runtime_assert.h>
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
	rassert(device.supports_cuda, 2026031419211100001);
	device.supports_opencl = false;
	device.supports_vulkan = false;

	rassert(device.printInfo(), 2026031419211100002);

	gpu::Context gpu_context;
	gpu_context.init(device.device_id_cuda);
	gpu_context.activate();
	return gpu_context;
}

} // namespace

TEST(cuda, aplusb)
{
#ifndef CUDA_SUPPORT
	GTEST_SKIP() << "CUDA support is disabled in this build";
#else
	std::vector<gpu::Device> devices = enumCUDADevices();
	if (devices.empty()) {
		GTEST_SKIP() << "No CUDA devices available";
	}

	unsigned int n = 10 * 1000 * 1000;

	std::vector<unsigned int> as(n, 0);
	std::vector<unsigned int> bs(n, 0);
	std::vector<unsigned int> cs(n, 0);
	for (size_t i = 0; i < n; ++i) {
		as[i] = 3 * (i + 5) + 7;
		bs[i] = 11 * (i + 13) + 17;
	}

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing CUDA device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateCUDAContext(devices[k]);

		gpu::gpu_mem_32u gpu_a, gpu_b, gpu_c;
		gpu_a.resizeN(n);
		gpu_b.resizeN(n);
		gpu_c.resizeN(n);

		gpu_a.writeN(as.data(), n);
		gpu_b.writeN(bs.data(), n);

		double rw_data_size = 3.0 * n * sizeof(as[0]);
		timer exec_timer;
		for (int iter = 0; iter < 3; ++iter) {
			exec_timer.restart();
			gpu::WorkSize worksize(256, n);
			cuda::aplusb(worksize, gpu_a, gpu_b, gpu_c, n);
			std::cout << "VRAM bandwidth: " << (rw_data_size / exec_timer.elapsed()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;

			gpu_c.readN(cs.data(), n);
			for (size_t i = 0; i < n; ++i) {
				rassert(as[i] + bs[i] == cs[i], 2026031419211100003);
			}
		}
	}
#endif
}
