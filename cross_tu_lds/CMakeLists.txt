cmake_minimum_required(VERSION 3.16.0)
project(cross_tu_lds)

add_executable(cross_tu_lds ../kernel_02.cpp ../kernel_03.cpp ../device_func_02.cpp cross_tu_lds.cpp)
