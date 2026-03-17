cmake_minimum_required(VERSION 3.5)

project(libutils)

set(HEADERS
        ../base/libbase/fast_random.h
        libutils/misc.h
        ../base/libbase/string_utils.h
        ../base/libbase/timer.h
        )

set(SOURCES
        libutils/misc.cpp
        ../base/libbase/string_utils.cpp
        )

option(GPU_CUDA_SUPPORT "CUDA support." OFF)

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (GPU_CUDA_SUPPORT)
    add_definitions(-DCUDA_SUPPORT)
endif()

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Threads::Threads libgpu)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
