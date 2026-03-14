#include <libgpu/context.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "kernels.h"

__global__ void aplusb_kernel(const unsigned int *a,
							  const unsigned int *b,
							  unsigned int *c,
							  unsigned int n)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= n)
		return;

	if (index == 0) {
		printf("CUDA printf test in aplusb.cu kernel! a[index]=%d b[index]=%d \n", a[index], b[index]);
	}

	curassert(3 * (index + 5) + 7 == a[index], 456234523);
	curassert(11 * (index + 13) + 17 == b[index], 657456342);

	c[index] = a[index] + b[index];
}

namespace cuda {
void aplusb(const gpu::WorkSize &workSize,
			const gpu::gpu_mem_32u &a,
			const gpu::gpu_mem_32u &b,
			gpu::gpu_mem_32u &c,
			unsigned int n)
{
	gpu::Context context;
	rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
	cudaStream_t stream = context.cudaStream();
	::aplusb_kernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), n);
	CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
