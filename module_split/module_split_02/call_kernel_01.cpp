#include "common.hpp"
#include "kernel_01.hpp"

void call_kernel_01() {
    std::vector<int> vx{1,2};
    std::vector<int> vy{2,3};
    std::vector<int> vz{100,101};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);
    hip_buffer<int> z(vz);

    kernel_01<<<1,vx.size()>>>(x.ptr(), y.ptr(), z.ptr());
    z.copy_from_buffer(vz);
    printf("kernel_01: ");
    for(auto i : vz) {
        printf("%d, ", i);
    }
    printf("\n");
}