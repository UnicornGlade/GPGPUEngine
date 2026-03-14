#pragma once

#include <libgpu/opencl/engine.h>

namespace ocl {
	ocl::ProgramBinaries &getAplusBKernel();
	ocl::ProgramBinaries &getFillBufferWithZerosKernel();
	ocl::ProgramBinaries &getRadixSort01LocalCountingKernel();
	ocl::ProgramBinaries &getRadixSort02GlobalPrefixesScanSumReductionKernel();
	ocl::ProgramBinaries &getRadixSort03GlobalPrefixesScanAccumulationKernel();
	ocl::ProgramBinaries &getRadixSort04ScatterKernel();
}
