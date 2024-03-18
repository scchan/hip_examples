#include "common.hpp"
#include "kernel_07.hpp"

__global__
void kernel_07(int* ax, int* ay, int* az) {
    auto i = hip_global_thread_id_1D();
    az[i] = ax[i] + ay[i];
}