#include "common.hpp"
#include "kernel_02.hpp"
#include "kernel_03.hpp"
#include "kernel_04.hpp"
#include "kernel_05.hpp"

#include <iostream>

void direct_call() {
    std::vector<int> vx{0,1,2,3};
    std::vector<int> vy{0,0,0,0};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);

    kernel_rotate_right<<<1,vx.size()>>>(x.ptr(), y.ptr(), 2);
    kernel_rotate_left<<<1,vx.size()>>>(y.ptr(), x.ptr(), 3);
    x.copy_from_buffer(vy);

    for(auto i : vy) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

void indirect_call() {
    std::vector<int> vx{0,1,2,3};
    std::vector<int> vy{0,0,0,0};

    hip_buffer<int> x(vx);
    hip_buffer<int> y(vy);
 
    init_wave_rotate_int_functions<<<1,1,1>>>();
   
    // rotate right
    kernel_rotate_int<<<1,vx.size()>>>(0, x.ptr(), y.ptr(), 2);
    // rotate left
    kernel_rotate_int<<<1,vx.size()>>>(1, y.ptr(), x.ptr(), 3);

    x.copy_from_buffer(vy);

    // should match the output from direct_call
    for(auto i : vy) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}


int main() {
    direct_call();
    indirect_call();
    return 0;
}