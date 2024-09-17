cmake_minimum_required(VERSION 3.16.0)
project(hip_examples)

set(HIP_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)

set(OPT_LEVEL -O3)
set(OFFLOAD_ARCH --offload-arch=gfx900 --offload-arch=gfx90a)
set(HIP_COMPILE_FLAGS -xhip ${OFFLOAD_ARCH})
set(RDC_FLAGS -fgpu-rdc)

add_compile_options(${HIP_COMPILE_FLAGS} ${RDC_FLAGS} ${OPT_LEVEL})

set(SPLIT_MODULE_FLAGS -Xoffload-linker --lto-partitions=16)
set(SPLIT_MODULE_FLAGS ${SPLIT_MODULE_FLAGS}  -mllvm --amdgpu-module-splitting-print-partition-summaries=module_split_summaries)

add_link_options(--hip-link ${OFFLOAD_ARCH} ${RDC_FLAGS} ${OPT_LEVEL} ${SPLIT_MODULE_FLAGS})

include_directories(.)
add_subdirectory(cross_tu_call)
add_subdirectory(cross_tu_lds)
add_subdirectory(indirect_call)
add_subdirectory(cross_tu_resource)
add_subdirectory(module_split_01)
add_subdirectory(module_split_02)