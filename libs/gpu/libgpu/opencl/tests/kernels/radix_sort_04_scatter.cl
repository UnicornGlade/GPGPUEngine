#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
	__global const uint* in,
	__global const uint* block_offsets,
	__global uint* out,
	__global uint* bin_base,
	const unsigned int n,
	const unsigned int shift)
{
	__local unsigned int shared_block_offsets[BINS_CNT];
	__local unsigned int shared_bin_base[BINS_CNT];
	__local unsigned int block_bins[GROUP_SIZE];

	const unsigned int global_ind = get_global_id(0);
	const unsigned int thread_ind = get_local_id(0);

	if (thread_ind < BINS_CNT) {
		shared_block_offsets[thread_ind] = block_offsets[(global_ind / GROUP_SIZE) * BINS_CNT + thread_ind];
		shared_bin_base[thread_ind] = bin_base[thread_ind];
	}

	const unsigned int val = (global_ind < n ? in[global_ind] : 0u);
	const unsigned int bin = (val >> shift) & RADIX_MASK;
	block_bins[thread_ind] = bin;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (global_ind < n) {
		unsigned int cnt = 0u;
		for (int i = thread_ind; i < GROUP_SIZE && global_ind - thread_ind + i < n; ++i) {
			if (block_bins[i] == bin) {
				++cnt;
			}
		}

		unsigned int index = shared_block_offsets[bin] - cnt;
		if (bin > 0u) {
			index += shared_bin_base[bin - 1u];
		}

		out[index] = val;
	}
}
