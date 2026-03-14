#pragma once

#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

namespace cuda {
void aplusb(const gpu::WorkSize &workSize,
	const gpu::gpu_mem_32u &a,
	const gpu::gpu_mem_32u &b,
	gpu::gpu_mem_32u &c,
	unsigned int n);
}
