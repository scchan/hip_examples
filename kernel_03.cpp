#include "common.hpp"
#include "device_func_02.hpp"
#include "kernel_03.hpp"

__global__
void kernel_rotate_left(int* ax, int* ay, unsigned int n) {
    auto i = hip_global_thread_id_1D();
    ay[i] = wave_rotate_lds_left(ax[i], n);
}

__global__
void kernel_rotate_left(double* ax, double* ay, unsigned int n) {
    auto i = hip_global_thread_id_1D();
    ay[i] = wave_rotate_lds_left(ax[i], n);
}