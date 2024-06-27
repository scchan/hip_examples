#include "common.hpp"
#include "kernel_04.hpp"
#include "kernel_05.hpp"

__global__
void kernel_rotate_int(unsigned int dir, int* ax, int* ay, unsigned int n) {
    if (dir < num_wr_funcs) {
      auto i = hip_global_thread_id_1D();
      auto t = wave_rotate_int_functions[dir](ax[i], n);
      //printf("i = %d, ax[i] = %d, ay[i] = %d \n", i, ax[i], t);
      ay[i] = t;
    }
}
