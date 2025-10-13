cmake_minimum_required(VERSION 3.15)
project(SoftmaxAcceleration LANGUAGES CXX CUDA)

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

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CUDA_SOURCES
    src/gpu/softmax_gpu.cu
)
add_library(cuda ${CUDA_SOURCES})
target_include_directories(cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(cuda PRIVATE -g)

find_package(dnnl QUIET)
if(dnnl_FOUND)
  message(STATUS "Found oneDNN, enabling oneDNN implementation.")
  target_sources(softmax PRIVATE src/softmax_onednn_cpu.cpp)
  target_link_libraries(softmax PUBLIC dnnl)
else()
  message(WARNING "oneDNN not found. oneDNN implementation will be disabled.")
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  add_library(softmax_omp
    src/cpu_parallel/softmax_naive_omp.cpp
    src/cpu_parallel/softmax_simd_omp.cpp
  )
  target_include_directories(softmax_omp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
  target_link_libraries(softmax_omp PUBLIC OpenMP::OpenMP_CXX)
  target_compile_options(softmax_omp PRIVATE $<$<CONFIG:Release>:-O3 -march=native -mavx2 -mfma>)
endif()

add_executable(bench benchmarks/main_bench.cpp)
target_link_libraries(bench
  common
  softmax
  cuda
  $<$<TARGET_EXISTS:softmax_omp>:softmax_omp>
  benchmark::benchmark
)

enable_testing()
add_executable(softmax_tests
  tests/softmax_tests.cpp
)
target_include_directories(softmax_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(softmax_tests
  common
  softmax
  $<$<TARGET_EXISTS:softmax_omp>:softmax_omp>
  GTest::gtest_main
)
if (TARGET softmax_omp)
  target_compile_definitions(softmax_tests PRIVATE HAVE_SOFTMAX_OMP=1)
  if (TARGET bench)
    target_compile_definitions(bench PRIVATE HAVE_SOFTMAX_OMP=1)
  endif()
endif()
if(dnnl_FOUND)
  target_compile_definitions(softmax_tests PRIVATE HAVE_ONEDNN=1)
  if (TARGET bench)
    target_compile_definitions(bench PRIVATE HAVE_ONEDNN=1)
  endif()
endif()
include(GoogleTest)
gtest_discover_tests(softmax_tests)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Project configured. Use the Makefile to build and run benchmarks.")
message(STATUS "Example: 'make benchmark' or 'make clean'")
