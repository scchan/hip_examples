cmake_minimum_required(VERSION 3.16.0)
project(cross_tu_call)

add_executable(cross_tu_call ../kernel_01.cpp ../device_func_01.cpp cross_tu_call.cpp)
