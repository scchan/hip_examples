#pragma once
#include "hip/hip_runtime.h"

extern
__device__
double high_vgpr_pressure(double*, const unsigned int);


extern
__global__
void same_module_calling_high_vgpr_pressure_func(double*, double*, const unsigned int);