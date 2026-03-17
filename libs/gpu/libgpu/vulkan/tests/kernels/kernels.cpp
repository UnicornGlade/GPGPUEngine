#include "kernels.h"

#include "generated_kernels/aplusb_comp.h"
#include "generated_kernels/atomic_add_comp.h"
#include "generated_kernels/batched_binary_search_comp.h"
#include "generated_kernels/fill_buffer_with_zeros_comp.h"
#include "generated_kernels/image_conversion_from_float_to_T_comp.h"
#include "generated_kernels/image_interpolation_comp.h"
#include "generated_kernels/radix_sort_01_local_counting_comp.h"
#include "generated_kernels/radix_sort_01_local_counting_portable_comp.h"
#include "generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_comp.h"
#include "generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_portable_comp.h"
#include "generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_comp.h"
#include "generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_portable_comp.h"
#include "generated_kernels/radix_sort_04_scatter_comp.h"
#include "generated_kernels/radix_sort_04_scatter_portable_comp.h"
#include "generated_kernels/rasterize_frag.h"
#include "generated_kernels/rasterize_vert.h"
#include "generated_kernels/rasterize_blending_frag.h"
#include "generated_kernels/write_value_at_index_comp.h"

#include <libgpu/vulkan/vk/common_host.h>

namespace avk2 {
	const ProgramBinaries& getAplusBKernel() {
		return vulkan_binaries_aplusb_comp;
	}
	const ProgramBinaries& getAtomicAddKernel() {
		return vulkan_binaries_atomic_add_comp;
	}
	const ProgramBinaries& getBatchedBinarySearch() {
		return vulkan_binaries_batched_binary_search_comp;
	}
	const ProgramBinaries& getFillBufferWithZerosKernel() {
		return vulkan_binaries_fill_buffer_with_zeros_comp;
	}
	const ProgramBinaries& getImageConversionFromFloatToT(DataType type) {
		rassert(type == DataType8u || type == DataType16u || type == DataType32f, 5462312324, typeName(type));
		std::string type_name = toupper(typeName(type));
		rassert(vulkan_binaries_image_conversion_from_float_to_T.count(type_name) > 0, 4613543412, type_name);
		return *vulkan_binaries_image_conversion_from_float_to_T[type_name];
	}
	const ProgramBinaries& getInterpolationKernel(DataType type, int nchannels) {
		rassert(type == DataType8u || type == DataType16u || type == DataType32f, 251615253, typeName(type));
		rassert(nchannels >= 1 && nchannels <= VK_MAX_NCHANNELS, 7348921634, nchannels);
		std::string type_name = to_string(nchannels) + "x" + toupper(typeName(type));
		rassert(vulkan_binaries_image_interpolation.count(type_name) > 0, 770673198, type_name);
		return *vulkan_binaries_image_interpolation[type_name];
	}
	const ProgramBinaries& getRadixSort01LocalCountingKernel() {
		return vulkan_binaries_radix_sort_01_local_counting_comp;
	}
	const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionKernel() {
		return vulkan_binaries_radix_sort_02_global_prefixes_scan_sum_reduction_comp;
	}
	const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationKernel() {
		return vulkan_binaries_radix_sort_03_global_prefixes_scan_accumulation_comp;
	}
	const ProgramBinaries& getRadixSort04ScatterKernel() {
		return vulkan_binaries_radix_sort_04_scatter_comp;
	}
	const ProgramBinaries& getRadixSort01LocalCountingPortableKernel() {
		return vulkan_binaries_radix_sort_01_local_counting_portable_comp;
	}
	const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReductionPortableKernel() {
		return vulkan_binaries_radix_sort_02_global_prefixes_scan_sum_reduction_portable_comp;
	}
	const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulationPortableKernel() {
		return vulkan_binaries_radix_sort_03_global_prefixes_scan_accumulation_portable_comp;
	}
	const ProgramBinaries& getRadixSort04ScatterPortableKernel() {
		return vulkan_binaries_radix_sort_04_scatter_portable_comp;
	}
	std::vector<const ProgramBinaries*> getRasterizeKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_frag};
	}
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_blending_frag};
	}
	const ProgramBinaries& getWriteValueAtIndexKernel() {
		return vulkan_binaries_write_value_at_index_comp;
	}
}
