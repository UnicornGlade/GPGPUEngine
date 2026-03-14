#pragma once

#include <libgpu/vulkan/engine.h>

namespace avk2 {
	const ProgramBinaries& getAplusBKernel();
	const ProgramBinaries& getAtomicAddKernel();
	const ProgramBinaries& getBatchedBinarySearch();
	const ProgramBinaries& getFillBufferWithZerosKernel();
	const ProgramBinaries& getImageConversionFromFloatToT(DataType type);
	const ProgramBinaries& getInterpolationKernel(DataType type, int nchannels);
	const ProgramBinaries& getRadixSort01LocalCountingKernel();
	const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionKernel();
	const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationKernel();
	const ProgramBinaries& getRadixSort04ScatterKernel();
	std::vector<const ProgramBinaries*> getRasterizeKernel();
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel();
	const ProgramBinaries& getWriteValueAtIndexKernel();
}
