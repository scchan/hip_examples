#include <bit>
#include "hsa/hsa.h"
#include "hip/hip_runtime.h"

void HIP_CHECK(hipError_t) {

}

__inline__
__device__ const void* get_kernarg_ptr() {
  return  std::bit_cast<const void*>(
    __builtin_amdgcn_kernarg_segment_ptr());
}

__inline__
__device__ const hsa_kernel_dispatch_packet_t* get_dispatch_ptr() {
  return  std::bit_cast<const hsa_kernel_dispatch_packet_t*>(
    __builtin_amdgcn_dispatch_ptr());
}

__inline__
__device__ const void* get_kernarg_ptr(const hsa_kernel_dispatch_packet_t* p) {
  return  p->kernarg_address;
}


__global__
void
dummy(const int* a, const int* b, const int* c) {
  
}


__global__
void
k_ptr(const int* a, const int* b, const int* c) {
  const void* kernarg_ptr = get_kernarg_ptr();
  const void* kernarg_ptr_from_dispatch = get_kernarg_ptr(get_dispatch_ptr());
  if (kernarg_ptr != kernarg_ptr_from_dispatch) {
    abort();
  }
  if (a != b || a != c) {
    abort();
  }
}

int main() {
    int* a = nullptr;
    int* b = nullptr;
    constexpr int num = 4096;
    HIP_CHECK(hipMalloc(&a, num * sizeof(int)));
    HIP_CHECK(hipMalloc(&b, num * sizeof(int)));
    dummy<<<1,1>>>(b,b,b);
    hipDeviceSynchronize();
    k_ptr<<<1024,64>>>(a, a, a);
    HIP_CHECK(hipFree(a));
    return 0;
}