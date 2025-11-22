cmake_minimum_required(VERSION 3.12)
project(deepx_core)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the Python extension module
pybind11_add_module(deepx_core cpp_module/geometry_engine.cpp)

# Optimization flags
if(MSVC)
    target_compile_options(deepx_core PRIVATE /O2 /W4)
else()
    target_compile_options(deepx_core PRIVATE -O3 -Wall -Wextra -march=native)
endif()

# Installation
install(TARGETS deepx_core DESTINATION .)