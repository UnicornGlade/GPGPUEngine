#pragma once

#include "utils.h"

#include "../../base/libbase/math.h"
#include "../../base/libbase/runtime_assert.h"

#ifdef CUDA_SUPPORT
	#include <vector_types.h>
#endif

namespace gpu {
	class WorkSize {
	public:
		WorkSize(size_t groupSizeX, size_t workSizeX)
		{
			init(1, groupSizeX, 1, 1, workSizeX, 1, 1);
		}

		WorkSize(size_t groupSizeX, size_t groupSizeY, size_t workSizeX, size_t workSizeY)
		{
			init(2, groupSizeX, groupSizeY, 1, workSizeX, workSizeY, 1);
		}

		WorkSize(size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ, size_t workSizeX, size_t workSizeY, size_t workSizeZ)
		{
			init(3, groupSizeX, groupSizeY, groupSizeZ, workSizeX, workSizeY, workSizeZ);
		}

#ifdef CUDA_SUPPORT
		const dim3 &cuBlockSize() const {
			return blockSize;
		}

		const dim3 &cuGridSize() const {
			return gridSize;
		}
#endif

		const size_t *clLocalSize() const {
			return localWorkSize;
		}

		const size_t *clGlobalSize() const {
			return globalWorkSize;
		}

		const size_t *vkGroupSize() const {
			return groupSize;
		}

		const size_t *vkGroupCount() const {
			return groupCount;
		}

		int clWorkDim() const {
			return workDims;
		}

		size_t workSizeX() const {
			return globalWorkSize[0];
		}

		size_t workSizeY() const {
			return globalWorkSize[1];
		}

		size_t workSizeZ() const {
			return globalWorkSize[2];
		}

	private:
		void init(int workDims, size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ, size_t workSizeX, size_t workSizeY, size_t workSizeZ)
		{
			// round up workSize so that it is divisible by groupSize
			workSizeX = div_ceil(workSizeX, groupSizeX) * groupSizeX;
			workSizeY = div_ceil(workSizeY, groupSizeY) * groupSizeY;
			workSizeZ = div_ceil(workSizeZ, groupSizeZ) * groupSizeZ;

			// Vulkan
			groupSize[0] = groupSizeX;
			groupSize[1] = groupSizeY;
			groupSize[2] = groupSizeZ;

			groupCount[0] = workSizeX / groupSizeX;
			groupCount[1] = workSizeY / groupSizeY;
			groupCount[2] = workSizeZ / groupSizeZ;

			// OpenCL
			this->workDims = workDims;

			localWorkSize[0] = groupSizeX;
			localWorkSize[1] = groupSizeY;
			localWorkSize[2] = groupSizeZ;

			globalWorkSize[0] = workSizeX;
			globalWorkSize[1] = workSizeY;
			globalWorkSize[2] = workSizeZ;

#ifdef CUDA_SUPPORT
			// CUDA
			blockSize	= dim3((unsigned int) groupSizeX, (unsigned int) groupSizeY, (unsigned int) groupSizeZ);
			gridSize	= dim3((unsigned int) groupCount[0], (unsigned int) groupCount[1], (unsigned int) groupCount[2]);
#endif
		}

	private:
		// Vulkan
		size_t	groupSize[3];
		size_t	groupCount[3];

		// OpenCL
		size_t	localWorkSize[3];
		size_t	globalWorkSize[3];
		int		workDims;

		// CUDA
#ifdef CUDA_SUPPORT
		dim3	blockSize;
		dim3	gridSize;
#endif
	};

	inline WorkSize WorkSize1DTo2D(size_t groupSizeX, size_t workSizeX, size_t maxGroupCountX = 65535)
	{
		rassert(groupSizeX > 0, 2026031519364500001);
		rassert(maxGroupCountX > 0, 2026031519364500002);

		size_t rounded_work_size_x = div_ceil(workSizeX, groupSizeX) * groupSizeX;
		size_t required_group_count_x = rounded_work_size_x / groupSizeX;
		size_t actual_group_count_x = std::min(required_group_count_x, maxGroupCountX);
		size_t actual_work_size_x = actual_group_count_x * groupSizeX;
		size_t actual_work_size_y = div_ceil(required_group_count_x, actual_group_count_x);
		return WorkSize(groupSizeX, 1, actual_work_size_x, actual_work_size_y);
	}
}
