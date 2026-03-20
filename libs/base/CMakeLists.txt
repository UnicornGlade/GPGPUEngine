cmake_minimum_required(VERSION 3.5)

project(libbase)

set(HEADERS
        libbase/data_type.h
        libbase/fast_random.h
        libbase/gtest_utils.h
        libbase/lru_cache.h
        libbase/math.h
        libbase/nvtx_markers.h
        libbase/omp_utils.h
        libbase/platform.h
        libbase/point.h
        libbase/runtime_assert.h
        libbase/string_utils.h
        libbase/thread_mutex.h
        libbase/timer.h
        )

set(SOURCES
        libbase/gtest_utils.cpp
        libbase/omp_utils.cpp
        libbase/string_utils.cpp
        libbase/thread_mutex.cpp
        )

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Threads::Threads ${GTEST_BOTH_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENMP_CXX_LINK_TARGET})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

if (GPGPU_ENABLE_NVTX_MARKERS)
    find_path(NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h
            HINTS
                /usr/local/cuda/targets/x86_64-linux/include
                /usr/local/cuda-13.0/targets/x86_64-linux/include
                /usr/local/cuda/nsight-systems/target-linux-x64/nvtx/include
                /usr/local/cuda-13.0/nsight-systems-2025.3.2/target-linux-x64/nvtx/include
                /usr/local/cuda-13.0/nsight-compute-2025.3.0/host/target-linux-x64/nvtx/include)
    if (NOT NVTX3_INCLUDE_DIR)
        message(FATAL_ERROR "GPGPU_ENABLE_NVTX_MARKERS=ON, but nvtx3/nvToolsExt.h was not found")
    endif ()
    target_include_directories(${PROJECT_NAME} PUBLIC ${NVTX3_INCLUDE_DIR})
endif ()
