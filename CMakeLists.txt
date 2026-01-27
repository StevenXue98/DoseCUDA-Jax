cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)

add_subdirectory(DoseCUDA/dose_kernels)

install(TARGETS dose_kernels DESTINATION ${SKBUILD_PROJECT_NAME})
