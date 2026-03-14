#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
	__global uint* inout,
	__global const uint* block_sums,
	const unsigned int n)
{
	__local unsigned int block_data[GROUP_SIZE];
	__local unsigned int block_data_copy[GROUP_SIZE];
	const unsigned int global_ind = get_global_id(0);
	const unsigned int thread_ind = get_local_id(0);

	block_data[thread_ind] = (global_ind < n ? inout[global_ind] : 0u);
	block_data_copy[thread_ind] = block_data[thread_ind];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (thread_ind & 16u) {
		block_data[thread_ind] += block_data_copy[thread_ind ^ 16u];
	}

	const unsigned int block_ind = global_ind >> WARP_LG;
	if (block_ind > 0u) {
		block_data[thread_ind] += block_sums[((block_ind - 1u) << (WARP_LG - 1u)) + (thread_ind & 15u)];
	}
	if (global_ind < n) {
		inout[global_ind] = block_data[thread_ind];
	}
}
