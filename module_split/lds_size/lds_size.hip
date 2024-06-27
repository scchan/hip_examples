
#include <bitset>
#include <iostream>
#include "hip/hip_runtime.h"

template <unsigned int lds_size>
__global__ void check_lds_size(unsigned int* r) {
    __shared__ unsigned char lds[lds_size];

    lds[hipThreadIdx_x%lds_size] = 0;
    if (hipThreadIdx_x > 4096) {
        // Add some bogus references to lds[] to make it stick.
        // This code should not be executed.
        lds[hipThreadIdx_x%*r]++;
        *r = lds[hipThreadIdx_x];
    }

    unsigned int v = 0;
    asm ("s_getreg_b32 %0, hwreg(HW_REG_LDS_ALLOC)"
         : "=r" (v));
    *r = v;
}

template <unsigned int lds_size>
void test() {
    unsigned int* r = nullptr;
    hipMalloc(&r, sizeof(unsigned int));
    unsigned int c = 0;
    hipMemcpy(r, &c, sizeof(unsigned int), hipMemcpyHostToDevice);
    check_lds_size<lds_size><<<1,1>>>(r);
    hipMemcpy(&c, r, sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipFree(r);

    std::cout << "lds size: " << lds_size << std::endl;
    std::cout << "\tHW_REG_LDS_ALLOC: 0x" <<  std::hex << c << std::dec << std::endl;
    std::cout << "\tHW_REG_LDS_ALLOC: " << std::bitset<32>{c} << std::endl;
}

int main() {
    test<64>();
    test<1024>();
    test<2048>();
    return 0;
}