cmake_minimum_required(VERSION 3.4...3.18)
project(render_utils VERSION 0.0.1)

add_subdirectory(third_party/pybind11)

pybind11_add_module(render_utils src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(render_utils
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})