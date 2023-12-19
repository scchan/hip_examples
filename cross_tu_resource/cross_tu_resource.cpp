#include "common.hpp"
#include "kernel_06.hpp"
#include <iostream>

int main() {
    constexpr auto n = 64;
    double *x, *y;
    cross_tu_calling_high_vgpr_pressume_func<<<1,n>>>(x,y,n);
    return 0;
}