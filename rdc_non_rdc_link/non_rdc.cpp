#include <cstdio>
#include "hip/hip_runtime.h"

extern "C"
__global__
void non_rdc_kernel(int* dummy) {
}

void call_dummy(int* d) {
   printf("call_dummy\n");
   non_rdc_kernel<<<1,1>>>(d);
}
