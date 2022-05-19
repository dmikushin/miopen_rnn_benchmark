cmake_minimum_required(VERSION 3.1)

project(miopen_rnn_benchmark)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)
find_package(miopen)

add_executable(miopen_rnn_benchmark src/miopen_rnn_benchmark.cpp)
target_include_directories(miopen_rnn_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# $(HIPCC) $< -o $@ -I./include -L/opt/rocm/miopen/lib -lMIOpen $(OPT) -std=c++11 --amdgpu-target=gfx900
set_property(TARGET miopen_rnn_benchmark PROPERTY CXX_STANDARD 11)
target_link_libraries(miopen_rnn_benchmark MIOpen hip::host)

