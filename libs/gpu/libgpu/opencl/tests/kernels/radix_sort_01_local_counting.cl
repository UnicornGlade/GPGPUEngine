#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
	__global const uint* in,
	__global uint* block_hist,
	const unsigned int shift,
	const unsigned int n)
{
	__local unsigned int bins[BINS_CNT];
	const unsigned int thread_ind = get_local_id(0);
	const unsigned int block_ind = get_group_id(0);
	const unsigned int global_ind = get_global_id(0);
	if (thread_ind < BINS_CNT) {
		bins[thread_ind] = 0u;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (global_ind < n) {
		const unsigned int bin = (in[global_ind] >> shift) & RADIX_MASK;
		atomic_inc(&bins[bin]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (thread_ind < BINS_CNT) {
		block_hist[thread_ind + block_ind * BINS_CNT] = bins[thread_ind];
	}
}
