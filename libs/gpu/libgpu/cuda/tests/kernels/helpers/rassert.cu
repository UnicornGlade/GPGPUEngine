#pragma once

#include <cstdio>

#define curassert(predicate, unique_code)                                                     \
	do {                                                                                      \
		if (!(predicate)) {                                                                   \
			printf("CUDA curassert failed: code=%lld\n", (long long)(unique_code));          \
			asm("trap;");                                                                     \
		}                                                                                     \
	} while (0)
