cmake_minimum_required(VERSION 3.16.0)
project(kernarg)

set(HIP_CXX_COMPILER "/opt/rocm/lib/llvm.org/bin/clang++")
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER} CACHE FILEPATH "" FORCE)

set(CMAKE_CXX_STANDARD 20)

set(OPT_LEVEL -O3)
set(OFFLOAD_ARCH --offload-arch=gfx900 --offload-arch=gfx90a --offload-arch=gfx950)
set(HIP_COMPILE_FLAGS -xhip ${OFFLOAD_ARCH})

add_compile_options(${HIP_COMPILE_FLAGS} ${RDC_FLAGS} ${OPT_LEVEL})
add_link_options(--hip-link ${OFFLOAD_ARCH} ${OPT_LEVEL})

add_executable(kernarg kernarg.cpp)