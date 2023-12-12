#include "device_func_01.hpp"

[[clang::noinline]]
__device__
int add(int x, int y) {
    return x + y;
}

