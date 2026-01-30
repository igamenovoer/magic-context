cmake_minimum_required(VERSION 3.18)
project(BuildCheck LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(SOURCES main.cpp test_basic.cu)
set(LIBRARIES CUDA::cudart)
set(DEFINITIONS "")

# --- Check for cuBLAS ---
find_package(CUDAToolkit)
if(TARGET CUDA::cublas)
    message(STATUS "Found cuBLAS, enabling test.")
    list(APPEND SOURCES test_cublas.cu)
    list(APPEND LIBRARIES CUDA::cublas)
    list(APPEND DEFINITIONS USE_CUBLAS)
endif()

# --- Check for cuDNN ---
# Note: FindCUDAToolkit in older CMake versions might not find cuDNN automatically 
# if it's in a separate location, but in Pixi envs (conda-forge/nvidia), it's usually 
# in the standard path or CUDAToolkit_ROOT.
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} $ENV{CONDA_PREFIX}/lib)
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIR} $ENV{CONDA_PREFIX}/include)

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    message(STATUS "Found cuDNN, enabling test.")
    list(APPEND SOURCES test_cudnn.cu)
    list(APPEND LIBRARIES ${CUDNN_LIBRARY})
    include_directories(${CUDNN_INCLUDE_DIR})
    list(APPEND DEFINITIONS USE_CUDNN)
endif()

# --- Check for NCCL ---
find_library(NCCL_LIBRARY nccl HINTS ${CUDAToolkit_LIBRARY_DIR} $ENV{CONDA_PREFIX}/lib)
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDAToolkit_INCLUDE_DIR} $ENV{CONDA_PREFIX}/include)

if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "Found NCCL, enabling test.")
    list(APPEND SOURCES test_nccl.cu)
    list(APPEND LIBRARIES ${NCCL_LIBRARY})
    include_directories(${NCCL_INCLUDE_DIR})
    list(APPEND DEFINITIONS USE_NCCL)
endif()

# --- Build Executable ---
add_executable(check_app ${SOURCES})
target_link_libraries(check_app PRIVATE ${LIBRARIES})
target_compile_definitions(check_app PRIVATE ${DEFINITIONS})
set_target_properties(check_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
