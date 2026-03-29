#include "kernels.h"

#include "generated_kernels/aplusb_comp.h"
#include "generated_kernels/atomic_add_comp.h"
#include "generated_kernels/batched_binary_search_comp.h"
#include "generated_kernels/brightness_stats_float_comp.h"
#include "generated_kernels/brightness_stats_rgb9e5_comp.h"
#include "generated_kernels/compare_hdr_rgb9e5_comp.h"
#include "generated_kernels/encode_bc6h_constant_comp.h"
#include "generated_kernels/fill_buffer_with_zeros_comp.h"
#include "generated_kernels/image_conversion_from_float_to_T_comp.h"
#include "generated_kernels/image_interpolation_comp.h"
#include "generated_kernels/jpeg_decode_color_420_comp.h"
#include "generated_kernels/jpeg_decode_color_420_coeffs_to_ycbcr_comp.h"
#include "generated_kernels/jpeg_decode_color_420_coeffs_to_ycbcr_colmeta_comp.h"
#include "generated_kernels/jpeg_decode_color_420_coeffs_to_ycbcr_int_comp.h"
#include "generated_kernels/jpeg_decode_color_420_to_coeffs_comp.h"
#include "generated_kernels/jpeg_decode_color_420_to_coeffs_colmeta_comp.h"
#include "generated_kernels/jpeg_decode_color_420_to_ycbcr_comp.h"
#include "generated_kernels/jpeg_decode_color_comp.h"
#include "generated_kernels/jpeg_decode_grayscale_comp.h"
#include "generated_kernels/radix_sort_01_local_counting_comp.h"
#include "generated_kernels/radix_sort_01_local_counting_portable_comp.h"
#include "generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_comp.h"
#include "generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_portable_comp.h"
#include "generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_comp.h"
#include "generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_portable_comp.h"
#include "generated_kernels/radix_sort_04_scatter_comp.h"
#include "generated_kernels/radix_sort_04_scatter_portable_comp.h"
#include "generated_kernels/pack_hdr_rgb9e5_comp.h"
#include "generated_kernels/rasterize_frag.h"
#include "generated_kernels/rasterize_hdr_gnome_frag.h"
#include "generated_kernels/rasterize_vert.h"
#include "generated_kernels/rasterize_blending_frag.h"
#include "generated_kernels/reduce_sum_packed_rgb8_to_u32_comp.h"
#include "generated_kernels/reduce_sum_u8_to_u32_comp.h"
#include "generated_kernels/reduce_sum_u32_to_u32_comp.h"
#include "generated_kernels/sift_match_bruteforce_local_comp.h"
#include "generated_kernels/sift_match_gemm_like_fp16_packed_comp.h"
#include "generated_kernels/sift_match_gemm_like_comp.h"
#include "generated_kernels/sift_match_gemm_like_vec4_comp.h"
#include "generated_kernels/sift_match_tensor_cores_comp.h"
#include "generated_kernels/sift_match_tensor_cores_nv_comp.h"
#include "generated_kernels/sift_match_tensor_cores_nv_preloaded_a_comp.h"
#include "generated_kernels/write_value_at_index_comp.h"
#include "generated_kernels/ycbcr420_to_rgb_comp.h"

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
	const ProgramBinaries& getBrightnessStatsFloatKernel() {
		return vulkan_binaries_brightness_stats_float_comp;
	}
	const ProgramBinaries& getBrightnessStatsRgb9e5Kernel() {
		return vulkan_binaries_brightness_stats_rgb9e5_comp;
	}
	const ProgramBinaries& getCompareHdrRgb9e5Kernel() {
		return vulkan_binaries_compare_hdr_rgb9e5_comp;
	}
	const ProgramBinaries& getEncodeBc6hConstantKernel() {
		return vulkan_binaries_encode_bc6h_constant_comp;
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
	const ProgramBinaries& getJpegDecodeColor420Kernel() {
		return vulkan_binaries_jpeg_decode_color_420_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrKernel() {
		return vulkan_binaries_jpeg_decode_color_420_coeffs_to_ycbcr_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrColmetaKernel() {
		return vulkan_binaries_jpeg_decode_color_420_coeffs_to_ycbcr_colmeta_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420CoeffsToYcbcrIntKernel() {
		return vulkan_binaries_jpeg_decode_color_420_coeffs_to_ycbcr_int_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420ToCoeffsKernel() {
		return vulkan_binaries_jpeg_decode_color_420_to_coeffs_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420ToCoeffsColmetaKernel() {
		return vulkan_binaries_jpeg_decode_color_420_to_coeffs_colmeta_comp;
	}
	const ProgramBinaries& getJpegDecodeColor420ToYcbcrKernel() {
		return vulkan_binaries_jpeg_decode_color_420_to_ycbcr_comp;
	}
	const ProgramBinaries& getJpegDecodeColorKernel() {
		return vulkan_binaries_jpeg_decode_color_comp;
	}
	const ProgramBinaries& getJpegDecodeGrayscaleKernel() {
		return vulkan_binaries_jpeg_decode_grayscale_comp;
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
	const ProgramBinaries& getPackHdrRgb9e5Kernel() {
		return vulkan_binaries_pack_hdr_rgb9e5_comp;
	}
	std::vector<const ProgramBinaries*> getRasterizeKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_frag};
	}
	std::vector<const ProgramBinaries*> getRasterizeHdrGnomeKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_hdr_gnome_frag};
	}
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_blending_frag};
	}
	const ProgramBinaries& getReduceSumPackedRgb8ToU32Kernel() {
		return vulkan_binaries_reduce_sum_packed_rgb8_to_u32_comp;
	}
	const ProgramBinaries& getReduceSumU8ToU32Kernel() {
		return vulkan_binaries_reduce_sum_u8_to_u32_comp;
	}
	const ProgramBinaries& getReduceSumU32ToU32Kernel() {
		return vulkan_binaries_reduce_sum_u32_to_u32_comp;
	}
	const ProgramBinaries& getSiftMatchBruteforceLocalKernel() {
		return vulkan_binaries_sift_match_bruteforce_local_comp;
	}
	const ProgramBinaries& getSiftMatchGemmLikeFp16PackedKernel() {
		return vulkan_binaries_sift_match_gemm_like_fp16_packed_comp;
	}
	const ProgramBinaries& getSiftMatchGemmLikeKernel() {
		return vulkan_binaries_sift_match_gemm_like_comp;
	}
	const ProgramBinaries& getSiftMatchGemmLikeVec4Kernel() {
		return vulkan_binaries_sift_match_gemm_like_vec4_comp;
	}
		const ProgramBinaries& getSiftMatchTensorCoresKernel() {
			return vulkan_binaries_sift_match_tensor_cores_comp;
		}
		const ProgramBinaries& getSiftMatchTensorCoresNvKernel() {
			return vulkan_binaries_sift_match_tensor_cores_nv_comp;
		}
		const ProgramBinaries& getSiftMatchTensorCoresNvPreloadedAKernel() {
			return vulkan_binaries_sift_match_tensor_cores_nv_preloaded_a_comp;
		}
		const ProgramBinaries& getWriteValueAtIndexKernel() {
			return vulkan_binaries_write_value_at_index_comp;
		}
	const ProgramBinaries& getYcbcr420ToRgbKernel() {
		return vulkan_binaries_ycbcr420_to_rgb_comp;
	}
}
