
#include "common.hpp"
#include "kernel_07.hpp"

void call_kernel_07() {
    std::vector<int> vx{1,2};
    std::vector<int> vy{2,3};
    std::vector<int> vz{100,101};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);
    hip_buffer<int> z(vz);

    kernel_07<<<1,vx.size()>>>(x.ptr(), y.ptr(), z.ptr());
    z.copy_from_buffer(vz);
    printf("kernel_07: ");
    for(auto i : vz) {
        printf("%d, ", i);
    }
    printf("\n");
}