#include "common.hpp"
#include "device_func_03.hpp"
#include "kernel_06.hpp"

__global__
#ifdef __LB_1024_1024__
__launch_bounds__(1024,1024)
#endif
void cross_tu_calling_high_vgpr_pressume_func(double* x, double* y, const unsigned int n) {
  y[hip_global_thread_id_1D()] = high_vgpr_pressure(x, n);
}