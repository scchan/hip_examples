cmake_minimum_required(VERSION 3.16.0)
project(indirect_call)

add_executable(indirect_call ../kernel_02.cpp ../kernel_03.cpp ../kernel_04.cpp 
                             ../device_func_02.cpp ../kernel_05.cpp indirect_call.cpp)
