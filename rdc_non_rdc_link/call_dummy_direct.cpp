#include "hip/hip_runtime.h"

extern "C"
__global__
void non_rdc_kernel(int* dummy);

int main() {
   int* d = nullptr;
   non_rdc_kernel<<<1,1>>>(d);
   return 0;
}
