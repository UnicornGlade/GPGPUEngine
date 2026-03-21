#pragma once

#include <libgpu/vulkan/engine.h>

namespace avk2 {
	const ProgramBinaries& getAplusBKernel();
	const ProgramBinaries& getAtomicAddKernel();
	const ProgramBinaries& getBatchedBinarySearch();
	const ProgramBinaries& getFillBufferWithZerosKernel();
	const ProgramBinaries& getImageConversionFromFloatToT(DataType type);
	const ProgramBinaries& getInterpolationKernel(DataType type, int nchannels);
	const ProgramBinaries& getJpegDecodeColor420Kernel();
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrKernel();
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrColmetaKernel();
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrIntKernel();
	const ProgramBinaries& getJpegDecodeColor420ToCoeffsKernel();
	const ProgramBinaries& getJpegDecodeColor420ToCoeffsColmetaKernel();
	const ProgramBinaries& getJpegDecodeColor420ToYcbcrKernel();
	const ProgramBinaries& getJpegDecodeColorKernel();
	const ProgramBinaries& getJpegDecodeGrayscaleKernel();
	const ProgramBinaries& getRadixSort01LocalCountingKernel();
	const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionKernel();
	const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationKernel();
	const ProgramBinaries& getRadixSort04ScatterKernel();
	const ProgramBinaries& getRadixSort01LocalCountingPortableKernel();
	const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionPortableKernel();
	const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationPortableKernel();
	const ProgramBinaries& getRadixSort04ScatterPortableKernel();
	std::vector<const ProgramBinaries*> getRasterizeKernel();
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel();
	const ProgramBinaries& getReduceSumPackedRgb8ToU32Kernel();
	const ProgramBinaries& getReduceSumU8ToU32Kernel();
	const ProgramBinaries& getReduceSumU32ToU32Kernel();
	const ProgramBinaries& getSiftMatchBruteforceLocalKernel();
	const ProgramBinaries& getSiftMatchGemmLikeFp16PackedKernel();
	const ProgramBinaries& getSiftMatchGemmLikeKernel();
	const ProgramBinaries& getSiftMatchGemmLikeVec4Kernel();
	const ProgramBinaries& getWriteValueAtIndexKernel();
	const ProgramBinaries& getYcbcr420ToRgbKernel();
}
