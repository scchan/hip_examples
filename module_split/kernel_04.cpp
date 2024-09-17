#include "kernel_04.hpp"

__device__
wr_int_fptr* wave_rotate_int_functions[num_wr_funcs];

__global__
void init_wave_rotate_int_functions() {
    if (hip_global_thread_id_1D() == 0) {
        wave_rotate_int_functions[0] = wave_rotate_lds_right;
        wave_rotate_int_functions[1] = wave_rotate_lds_left;
    }
}

__device__
math_int_fptr* math_int_functions[num_math_funcs];

__global__
void init_math_int_functions() {
    if (hip_global_thread_id_1D() == 0) {
        math_int_functions[0] = add;
        math_int_functions[1] = sub;
        math_int_functions[2] = mul;
    }
}
