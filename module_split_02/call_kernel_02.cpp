#include "common.hpp"
#include "kernel_02.hpp"
#include <iostream>

void call_kernel_02() {
    std::vector<int> vx{0,1,2,3};
    std::vector<int> vy{0,0,0,0};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);

    kernel_rotate_right<<<1,vx.size()>>>(x.ptr(), y.ptr(), 2);
    y.copy_from_buffer(vy);

    for(auto i : vy) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}