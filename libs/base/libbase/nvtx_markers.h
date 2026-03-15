#pragma once

#include <cstdint>

#ifdef GPGPU_ENABLE_NVTX_MARKERS
#include <nvtx3/nvToolsExt.h>
#endif

namespace profiling {

class ScopedRange {
public:
	explicit ScopedRange(const char *name, std::uint32_t argb = 0xFF4DA3FF)
	{
#ifdef GPGPU_ENABLE_NVTX_MARKERS
		nvtxEventAttributes_t attributes{};
		attributes.version = NVTX_VERSION;
		attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		attributes.colorType = NVTX_COLOR_ARGB;
		attributes.color = argb;
		attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
		attributes.message.ascii = name;
		nvtxRangePushEx(&attributes);
#else
		(void) name;
		(void) argb;
#endif
	}

	~ScopedRange()
	{
#ifdef GPGPU_ENABLE_NVTX_MARKERS
		nvtxRangePop();
#endif
	}

	ScopedRange(const ScopedRange &) = delete;
	ScopedRange &operator=(const ScopedRange &) = delete;
};

} // namespace profiling

