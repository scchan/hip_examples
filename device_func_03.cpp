#include "common.hpp"
#include "device_func_03.hpp"
#include <cmath>

[[clang::always_inline]]
static __device__
double high_vgpr_pressure(double x) {
    return sin(x) + exp(x) + rsqrt(x) + logb(x) + normcdf(x);
}


[[clang::noinline]]
__device__
double high_vgpr_pressure(double* d, const unsigned int n) {
    constexpr auto m = 16;
    double t[m];
    auto i = hip_global_thread_id_1D();

    #pragma unroll m
    for (auto j = 0; j < m; ++j) {
        t[j] = d[(i + j *  blockDim.x)%n];
    }

    double a = 0.0;
    #pragma unroll m
    for (auto j = 0; j < m; ++j) {
        a+=high_vgpr_pressure(t[j]);
    }
    return a;
}

// A dummy kernel (with low launch bounds) is used trying to persuade the compiler
// that the high_vgpr_pressure function should and is allowed to use more VGPRs
__global__
__launch_bounds__(256,256)  
void same_module_calling_high_vgpr_pressure_func(double* d, const unsigned int n) {
    auto i = hip_global_thread_id_1D();
    d[i] = high_vgpr_pressure(d,n);
}

