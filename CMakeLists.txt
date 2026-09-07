cmake_minimum_required(VERSION 3.18)
project(dfa_sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Target sm_80 (A100) and newer
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

option(USE_OPENMP "Enable OpenMP for host-side DFA compilation" ON)
if(USE_OPENMP)
    find_package(OpenMP)
endif()

# Compiler flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda")

# Static library containing DFA compilation and kernel code
add_library(dfa_sample_lib STATIC
    src/dfa_compile.cu
    src/dfa_mask.cu
    src/fused_sample.cu
)

target_include_directories(dfa_sample_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(dfa_sample_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(dfa_sample_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=-v>
)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(dfa_sample_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(dfa_sample_lib PUBLIC USE_OPENMP)
endif()

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE dfa_sample_lib CUDA::curand)
target_include_directories(test_correctness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Sequence generation test
add_executable(test_sequence tests/test_sequence.cu)
target_link_libraries(test_sequence PRIVATE dfa_sample_lib CUDA::curand)
target_include_directories(test_sequence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(test_sequence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency PRIVATE dfa_sample_lib CUDA::curand)
target_include_directories(bench_latency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CTest integration
enable_testing()
add_test(NAME correctness COMMAND test_correctness)
add_test(NAME sequence COMMAND test_sequence)
