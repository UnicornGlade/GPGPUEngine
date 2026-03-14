#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
	__global const uint* in,
	__global uint* out,
	const unsigned int src_size,
	const unsigned int dst_size)
{
	const unsigned int global_ind = get_global_id(0);
	const unsigned int thread_ind = global_ind & 31u;
	__local unsigned int bins[BINS_CNT];
	__local unsigned int prefix[BINS_CNT];

	if (global_ind < dst_size) {
		const unsigned int base = ((global_ind & 0xfffffff0u) << 1);
		const unsigned int val1 = (base + thread_ind < src_size ? in[base + thread_ind] : 0u);
		const unsigned int val2 = (base + (thread_ind ^ 16u) < src_size ? in[base + (thread_ind ^ 16u)] : 0u);
		if (dst_size > BINS_CNT) {
			out[global_ind] = val1 + val2;
		} else {
			bins[global_ind] = val1 + val2;
			prefix[global_ind] = 0u;
			int i = global_ind;
			while (i >= 0) {
				prefix[global_ind] += bins[i];
				--i;
			}
			out[global_ind] = prefix[global_ind];
		}
	}
}
