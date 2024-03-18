cmake_minimum_required(VERSION 3.16.0)
project(module_split_01)

add_executable(module_split_01 ../kernel_01.cpp ../device_func_01.cpp ../kernel_07.cpp module_split_01.cpp)
