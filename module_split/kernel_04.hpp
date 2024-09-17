#pragma once
#include <type_traits>
#include <functional>

#include "hip/hip_runtime.h"
#include "common.hpp"
#include "device_func_01.hpp"
#include "device_func_02.hpp"

typedef int(wr_int_fptr)(int, unsigned int);
constexpr auto num_wr_funcs = 2;
extern 
__device__ wr_int_fptr* wave_rotate_int_functions[num_wr_funcs];
__global__
void init_wave_rotate_int_functions();


typedef int(math_int_fptr)(int, int);
constexpr auto num_math_funcs = 3;
extern 
__device__ math_int_fptr* math_int_functions[num_math_funcs];
__global__
void init_math_int_functions();

