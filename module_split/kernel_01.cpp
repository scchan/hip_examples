#include "common.hpp"
#include "device_func_01.hpp"
#include "kernel_01.hpp"

__global__
void kernel_01(int* ax, int* ay, int* az) {
    auto i = hip_global_thread_id_1D();
    az[i] = add(ax[i],ay[i]);
}