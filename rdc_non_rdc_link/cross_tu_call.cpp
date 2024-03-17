#include "common.hpp"
#include "kernel_01.hpp"

#if 0
extern "C"
__global__
void non_rdc_kernel(int*);
#endif

extern
void call_dummy(int*);

int main() {
    std::vector<int> vx{1,2};
    std::vector<int> vy{2,3};
    std::vector<int> vz{100,101};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);
    hip_buffer<int> z(vz);
    kernel_01<<<1,vx.size()>>>(x.ptr(), y.ptr(), z.ptr());

    int* d = nullptr;
    call_dummy(d);
    //non_rdc_kernel<<<1,1>>>(d);

    z.copy_from_buffer(vz);
    for(auto i : vz) {
        printf("%d, ", i);
    }
    printf("\n");
    return 0;
}
