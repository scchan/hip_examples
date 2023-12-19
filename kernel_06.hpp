#pragma once
#include "hip/hip_runtime.h"
__global__
void cross_tu_calling_high_vgpr_pressume_func(double*, double*, const unsigned int);