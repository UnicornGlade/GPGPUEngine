#include <gtest/gtest.h>

#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/shared_device_buffer.h>

#include <cstdlib>

#include "test_utils.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"

namespace {

struct WriteValueAtIndexParams {
	unsigned int chosenBuffer;
	int index;
	unsigned int value;
};

bool areTimestampQueriesEnabled()
{
	const char *value = std::getenv("GPGPU_ENABLE_VULKAN_TIMESTAMP_QUERIES");
	if (value == nullptr) {
		return true;
	}
	std::string normalized = tolower(trimmed(value));
	return !(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no");
}

} // namespace

TEST(vulkan, asyncTinyWriteValueChain)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.setMemoryGuardsChecksAfterKernels(false);

		avk2::KernelSource kernel_write_value(avk2::getWriteValueAtIndexKernel());
		const int nlaunches = 1024;
		gpu::gpu_mem_32u buffer0(nlaunches / 2);
		gpu::gpu_mem_32u buffer1(nlaunches / 2);
		buffer0.fill(0);
		buffer1.fill(0);

		context.vk()->setMaxInflightComputeLaunches(32);
		context.vk()->resetAsyncComputeStats();
		timer timer_default;
		for (int launch = 0; launch < nlaunches; ++launch) {
			WriteValueAtIndexParams params{static_cast<unsigned int>(launch & 1), launch / 2, static_cast<unsigned int>(launch + 7)};
			kernel_write_value.exec(params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, 1), buffer0, buffer1);
		}
		double elapsed_default = timer_default.elapsed();
		std::cout << "asyncTinyWriteValueChain default launches/sec: " << (nlaunches / elapsed_default) << std::endl;
		context.vk()->logAsyncComputeStats("asyncTinyWriteValueChain/default");

		std::vector<unsigned int> got0 = buffer0.readVector();
		std::vector<unsigned int> got1 = buffer1.readVector();
		for (int i = 0; i < nlaunches / 2; ++i) {
			EXPECT_EQ(got0[i], static_cast<unsigned int>(2 * i + 7));
			EXPECT_EQ(got1[i], static_cast<unsigned int>(2 * i + 8));
		}

		buffer0.fill(0);
		buffer1.fill(0);
		context.vk()->setMaxInflightComputeLaunches(2);
		context.vk()->resetAsyncComputeStats();
		timer timer_limited;
		for (int launch = 0; launch < nlaunches; ++launch) {
			WriteValueAtIndexParams params{static_cast<unsigned int>(launch & 1), launch / 2, static_cast<unsigned int>(launch + 17)};
			kernel_write_value.exec(params, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, 1), buffer0, buffer1);
		}
		double elapsed_limited = timer_limited.elapsed();
		std::cout << "asyncTinyWriteValueChain inflight=2 launches/sec: " << (nlaunches / elapsed_limited) << std::endl;
		avk2::AsyncComputeStats limited_stats = context.vk()->getAsyncComputeStats(true);
		context.vk()->logAsyncComputeStats("asyncTinyWriteValueChain/maxInflight=2", false);
		if (areTimestampQueriesEnabled()) {
			EXPECT_GT(limited_stats.total_seconds, 0.0);
		} else {
			EXPECT_EQ(limited_stats.total_seconds, 0.0);
		}

		got0 = buffer0.readVector();
		got1 = buffer1.readVector();
		for (int i = 0; i < nlaunches / 2; ++i) {
			EXPECT_EQ(got0[i], static_cast<unsigned int>(2 * i + 17));
			EXPECT_EQ(got1[i], static_cast<unsigned int>(2 * i + 18));
		}

		context.vk()->setMaxInflightComputeLaunches(32);
	}

	checkPostInvariants();
}

TEST(vulkan, asyncAplusbChain)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	const unsigned int n = 1u << 18;
	const int nlaunches = 16;

	std::vector<unsigned int> as(n);
	std::vector<unsigned int> bs(n);
	for (size_t i = 0; i < n; ++i) {
		as[i] = 3 * (i + 5) + 7;
		bs[i] = 11 * (i + 13) + 17;
	}

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
		context.setMemoryGuardsChecksAfterKernels(false);

		avk2::KernelSource kernel_aplusb(avk2::getAplusBKernel());
		gpu::gpu_mem_32u gpu_a(n), gpu_b(n), gpu_c0(n), gpu_c1(n);
		gpu_a.writeN(as.data(), n);
		gpu_b.writeN(bs.data(), n);

		context.vk()->resetAsyncComputeStats();
		timer timer_chain;
		for (int launch = 0; launch < nlaunches; ++launch) {
			gpu::gpu_mem_32u &out = (launch & 1) ? gpu_c1 : gpu_c0;
			kernel_aplusb.exec(n, gpu::WorkSize1DTo2D(VK_GROUP_SIZE, n), gpu_a, gpu_b, out);
		}
		double elapsed = timer_chain.elapsed();
		double total_rw_gb = (3.0 * n * sizeof(unsigned int) * nlaunches) / (1024.0 * 1024.0 * 1024.0);
		std::cout << "asyncAplusbChain effective bandwidth: " << (total_rw_gb / elapsed) << " GB/s" << std::endl;
		context.vk()->logAsyncComputeStats("asyncAplusbChain");

		std::vector<unsigned int> got0 = gpu_c0.readVector();
		std::vector<unsigned int> got1 = gpu_c1.readVector();
		for (size_t i = 0; i < n; ++i) {
			unsigned int expected = bs[i] + as[i];
			EXPECT_EQ(got0[i], expected);
			EXPECT_EQ(got1[i], expected);
		}
	}

	checkPostInvariants();
}
