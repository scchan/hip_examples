#pragma once
#include "hip/hip_runtime.h"

__global__
void kernel_rotate_left(int*, int*, unsigned int);

__global__
void kernel_rotate_left(double*, double*, unsigned int);
