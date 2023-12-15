#include "common.hpp"
#include "device_func_02.hpp"
#include "kernel_02.hpp"

__global__
void kernel_rotate_right(int* ax, int* ay, unsigned int n) {
    auto i = hip_global_thread_id_1D();
    ay[i] = wave_rotate_lds_right(ax[i], n);
}
__global__
void kernel_rotate_right(double* ax, double* ay, unsigned int n) {
    auto i = hip_global_thread_id_1D();
    ay[i] = wave_rotate_lds_right(ax[i], n);
}