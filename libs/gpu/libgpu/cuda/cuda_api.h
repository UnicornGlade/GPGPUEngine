#pragma once

#include <libgpu/cuda/utils.h>
#include <cuda.h>

bool cuda_api_init();
CUresult CUDAAPI cuDeviceGetDynamic(CUdevice *device, int ordinal);
CUresult CUDAAPI cuCtxDestroyDynamic(CUcontext ctx);
CUresult CUDAAPI cuDeviceGetAttributeDynamic(int *pi, CUdevice_attribute attrib, CUdevice dev);

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
CUresult CUDAAPI cuCtxCreateDynamic(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev);
#else
CUresult CUDAAPI cuCtxCreateDynamic(CUcontext *pctx, unsigned int flags, CUdevice dev);
#endif

namespace cuda {

	std::string formatDriverError(CUresult code);

	static inline void reportErrorCU(CUresult err, int line, std::string prefix="")
	{
		if (CUDA_SUCCESS == err)
			return;

		std::string message = prefix + formatDriverError(err) + " at line " + to_string(line);

		switch (err) {
		case CUDA_ERROR_OUT_OF_MEMORY:
			throw cuda_bad_alloc(message);
		default:
			throw cuda_exception(message);
		}
	}

	#define CU_SAFE_CALL(expr)  cuda::reportErrorCU(expr, __LINE__)

}
