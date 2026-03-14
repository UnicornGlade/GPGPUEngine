#ifndef defines_vk // pragma once
#define defines_vk

#define BLOCK_THREADS		256
#define VK_GROUP_SIZE		BLOCK_THREADS
#define VK_GROUP_SIZE_X		16
#define VK_GROUP_SIZE_Y		16
#define BITS_AT_A_TIME		4
#define BINS_CNT			16
#define NUM_BOXES			BINS_CNT
#define RADIX_MASK			(BINS_CNT - 1)
#define BITS_IN_RADIX_SORT_ITERATION BITS_AT_A_TIME
#define WARP_LG				5

#define SPECIAL_EMPTY_VALUE	239
#define SOME_COLOR_VALUE	23 
#define MAX_F32_USED_VALUE	1000000.0f

#define BLENDING_RED_VALUE		0.5f
#define BLENDING_GREEN_VALUE	1.0f
#define BLENDING_BLUE_VALUE		(-2.0f)
#define BLENDING_ALPHA_VALUE	128.0f

#endif // pragma once
