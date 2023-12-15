#include "device_func_02.hpp"

static constexpr unsigned int wave_size = 64;

template <typename T>
__device__
T __wave_rotate_lds_right(T v, unsigned int n) {
    __shared__ T t[wave_size];
    T r = v;
    if (threadIdx.x < wave_size){
        t[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < wave_size){
        r = t[(threadIdx.x + blockDim.x - n) % blockDim.x];
    }
    return r;
}

[[clang::noinline]]
__device__
int wave_rotate_lds_right(int v, unsigned int n) {
    return __wave_rotate_lds_right(v, n);
}
[[clang::noinline]]
__device__
double wave_rotate_lds_right(double v, unsigned int n) {
    return __wave_rotate_lds_right(v, n);
}



template <typename T>
__device__
T __wave_rotate_lds_left(T v, unsigned int n) {
    return wave_rotate_lds_right(v, blockDim.x - n % blockDim.x); 
}

[[clang::noinline]]
__device__
int wave_rotate_lds_left(int v, unsigned int n) {
    return __wave_rotate_lds_left(v, n); 
}
[[clang::noinline]]
__device__
double wave_rotate_lds_left(double v, unsigned int n) {
    return __wave_rotate_lds_left(v, n); 
}



