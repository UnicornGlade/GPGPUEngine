#include "kernels.h"

#include "generated_kernels/aplusb.h"
#include "generated_kernels/fill_buffer_with_zeros.h"
#include "generated_kernels/radix_sort_01_local_counting.h"
#include "generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction.h"
#include "generated_kernels/radix_sort_03_global_prefixes_scan_accumulation.h"
#include "generated_kernels/radix_sort_04_scatter.h"

namespace ocl {
	ProgramBinaries& getAplusBKernel() {
		return opencl_binaries_aplusb;
	}
	ProgramBinaries& getFillBufferWithZerosKernel() {
		return opencl_binaries_fill_buffer_with_zeros;
	}
	ProgramBinaries& getRadixSort01LocalCountingKernel() {
		return opencl_binaries_radix_sort_01_local_counting;
	}
	ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionKernel() {
		return opencl_binaries_radix_sort_02_global_prefixes_scan_sum_reduction;
	}
	ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationKernel() {
		return opencl_binaries_radix_sort_03_global_prefixes_scan_accumulation;
	}
	ProgramBinaries& getRadixSort04ScatterKernel() {
		return opencl_binaries_radix_sort_04_scatter;
	}
}
