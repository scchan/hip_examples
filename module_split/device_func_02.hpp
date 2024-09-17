#pragma once
#include "hip/hip_runtime.h"

extern
__device__
int wave_rotate_lds_right(int,unsigned int);

extern
__device__
int wave_rotate_lds_left(int,unsigned int);

extern
__device__
double wave_rotate_lds_right(double,unsigned int);

extern
__device__
double wave_rotate_lds_left(double,unsigned int);
