cmake_minimum_required(VERSION 3.16.0)
project(cross_tu_resource)

add_executable(cross_tu_resource ../kernel_06.cpp ../device_func_03.cpp cross_tu_resource.cpp)

set_source_files_properties(../kernel_06.cpp PROPERTIES COMPILE_DEFINITIONS __K6_LB_1024_1024__)
