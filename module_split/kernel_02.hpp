#pragma once
#include "hip/hip_runtime.h"

__global__
void kernel_rotate_right(int*, int*, unsigned int);

__global__
void kernel_rotate_right(double*, double*, unsigned int);