#pragma once
#include "hip/hip_runtime.h"
__global__
void kernel_rotate_int(unsigned int, int*, int*, unsigned int);