#pragma once
#include <type_traits>
#include <functional>

#include "hip/hip_runtime.h"
#include "common.hpp"
#include "device_func_02.hpp"

typedef int(wr_int_fptr)(int, unsigned int);


extern 
__device__ wr_int_fptr* wave_rotate_int_functions[2];

__global__
void init_wave_rotate_int_functions();
