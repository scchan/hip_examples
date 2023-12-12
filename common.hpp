#pragma once
#include "hip/hip_runtime.h"
#include <algorithm>
#include <vector>

[[clang::always_inline]]
static
__device__
auto hip_global_thread_id_1D() {
    // 1D grid and 1D blocks
    return blockDim.x * blockIdx.x + threadIdx.x;
}

template<typename T>
class hip_buffer {
public:
    hip_buffer(size_t num) : n(num) {
        hipMalloc(&p, n * sizeof(T));
    };
    ~hip_buffer() {
        hipFree(p);
    }
    static constexpr size_t __type_size = sizeof(T);
    typedef std::vector<T> __V;
    hip_buffer(const __V& v) : hip_buffer(v.size()) {
        copy_to_buffer(v);
    }

    T* ptr() { return p; }

    auto copy_to_buffer(const __V& v) {
        auto s = std::min(n, v.size());
        hipMemcpy(p, v.data(), s * __type_size, hipMemcpyHostToDevice);
        return s;
    }
    auto copy_from_buffer(__V& v) {
        auto s = std::min(n, v.size());
        hipMemcpy(v.data(), p, s * __type_size, hipMemcpyDeviceToHost);
        return s;
    }
private:
    const size_t n;
    T* p;
};