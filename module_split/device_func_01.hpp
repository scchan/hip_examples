#pragma once
#include "hip/hip_runtime.h"

extern "C" {

__device__
int add(int,int);

__device__
int sub(int,int);

__device__
int mul(int,int);

}