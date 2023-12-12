#include "common.hpp"
#include "kernel_01.hpp"

int main() {
    std::vector<int> vx{1,2};
    std::vector<int> vy{2,3};
    std::vector<int> vz{100,101};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);
    hip_buffer<int> z(vz);
    kernel_01<<<vx.size(),1,1>>>(x.ptr(), y.ptr(), z.ptr());

    z.copy_from_buffer(vz);
    for(auto i : vz) {
        printf("%d, ", i);
    }
    printf("\n");
    return 0;
}