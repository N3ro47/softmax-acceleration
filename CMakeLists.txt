cmake_minimum_required(VERSION 3.15)
project(SoftmaxAcceleration LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto -mavx2 -mfma")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG    v1.9.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_library(common
  common/utils.cpp
  common/utils.h
)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(SOFTMAX_SOURCES
  src/cpu_sequential/softmax_naive_cpu.cpp
  src/cpu_sequential/softmax_foolish_handcoding_cpu.cpp
  src/cpu_sequential/softmax_simd_cpu.cpp
  src/cpu_sequential/softmax_fused_simd_cpu.cpp
)
add_library(softmax ${SOFTMAX_SOURCES})
target_include_directories(softmax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(bench benchmarks/main_bench.cpp)
target_link_libraries(bench
  common
  softmax
  benchmark::benchmark
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Project configured. Use the Makefile to build and run benchmarks.")
message(STATUS "Example: 'make benchmark' or 'make clean'")
