#include "device_func_01.hpp"

[[clang::noinline]]
__device__
int add(int x, int y) {
    return x + y;
}

[[clang::noinline]]
__device__
int sub(int x, int y) {
    return x + y;
}

[[clang::noinline]]
__device__
int mul(int x, int y) {
    return x * y;
}

