cmake_minimum_required(VERSION 3.16.0)
project(hip_examples)

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)

set(OFFLOAD_ARCH --offload-arch=gfx900)
set(HIP_COMPILE_FLAGS -xhip ${OFFLOAD_ARCH})
set(RDC_FLAGS -fgpu-rdc)

add_compile_options(${HIP_COMPILE_FLAGS} ${RDC_FLAGS})
add_link_options(--hip-link ${OFFLOAD_ARCH} ${RDC_FLAGS})

include_directories(.)
add_subdirectory(cross_tu_call)
