cmake_minimum_required(VERSION 3.16.0)
project(module_split_02)

add_executable(module_split_02 ../kernel_01.cpp ../device_func_01.cpp ../kernel_02.cpp ../device_func_02.cpp ../kernel_07.cpp module_split_02.cpp call_kernel_01.cpp call_kernel_07.cpp call_kernel_02.cpp)
