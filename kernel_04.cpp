#include "kernel_04.hpp"

__device__
wr_int_fptr* wave_rotate_int_functions[2];

__global__
void init_wave_rotate_int_functions() {
    if (hip_global_thread_id_1D() == 0) {
        wave_rotate_int_functions[0] = wave_rotate_lds_right;
        wave_rotate_int_functions[1] = wave_rotate_lds_left;
    }
}
