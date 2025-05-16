cmake_minimum_required(VERSION 3.14)
project(ODESolver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(INSTALL_GTEST OFF)
set(gtest_build_tests OFF)
# --- GoogleTest ---
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

#--- Google Benchmark ---
set(BENCHMARK_ENABLE_TESTING        OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(benchmark)

add_library(core INTERFACE)

target_include_directories(core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/structures>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/odesolver>
)

add_executable(${PROJECT_NAME}
  main.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE
    -O3
    -pipe
    -flto
    -march=native
    -fopenmp-simd
    -ffast-math
    -ffp-contract=fast
    -fomit-frame-pointer
  )
endif()


target_link_libraries(${PROJECT_NAME} PRIVATE core)
