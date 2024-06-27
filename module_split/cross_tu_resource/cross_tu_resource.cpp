#include "common.hpp"
#include "device_func_03.hpp"
#include "kernel_06.hpp"

#include <algorithm>
#include <iostream>

int main() {
    constexpr auto n = 64;
    std::vector<double> vx(n);
    std::generate(vx.begin(), vx.end(), []{ static auto t = 1.0; return t++; });

    std::vector<double> vy0(n, 0.0);
    std::vector<double> vy1(n, 0.0);

    // Here we have two kernels calling the same device function that has high register pressure.
    // We need to ensure that the kernel that's making a call crossing a TU boundary gets the
    // correct resource reporting (in this case, register budget) in RDC mode.

    hip_buffer<double> gx(vx);
    hip_buffer<double> gy(vy0);
    same_module_calling_high_vgpr_pressure_func<<<1, n>>>(gx.ptr(), gy.ptr(), n);
    gy.copy_from_buffer(vy0);

    gy.copy_to_buffer(vy1);
    cross_tu_calling_high_vgpr_pressume_func<<<1, n>>>(gx.ptr(), gy.ptr(), n);
    gy.copy_from_buffer(vy1);

    int errors = 0;
    for (auto i = 0; i < n; ++i) {
        if (vy0[i] != vy1[i]) {
            std::cerr << "Element " << i << " : vy0=" << vy0[i] << ", vy1=" << vy1[i] <<std::endl;
            ++errors;
        }
    }
    std::cout << errors << " Errors" << std::endl;

    return 0;
}