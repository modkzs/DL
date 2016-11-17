cmake_minimum_required(VERSION 3.6)
project(DL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp linear_arg.h)
add_executable(DL ${SOURCE_FILES})