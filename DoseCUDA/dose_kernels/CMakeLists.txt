set(MODULE_NAME dose_kernels)

Python_add_library(${MODULE_NAME} MODULE)

target_sources(${MODULE_NAME}
    PRIVATE
        dosemodule.cu
        CudaClasses.cu
        IMPTClasses.cu
        IMRTClasses.cu
        TextureClasses.cu)

target_include_directories(${MODULE_NAME}
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(${MODULE_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE   ON
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(${MODULE_NAME} PRIVATE Python::NumPy)
