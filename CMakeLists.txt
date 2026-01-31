cmake_minimum_required(VERSION 3.18)
project(CUDA_KERNELS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# find cudnn
set(CUDNN_ROOT "" CACHE PATH "Path to cuDNN installation")
if(NOT CUDNN_ROOT)
    set(CUDNN_ROOT ${CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_DIR}/..)
endif()

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${CUDNN_ROOT}
    PATH_SUFFIXES include
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${CUDNN_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Please set CUDNN_ROOT to the root directory of cuDNN 
    (e.g., /usr/local/cuda or your extracted folder).")
endif()

add_library(cuDNN::cuDNN UNKNOWN IMPORTED)
set_target_properties(cuDNN::cuDNN PROPERTIES
    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
)

macro(set_optimization_flags TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3,-v>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )
endmacro()

# output to ./bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set ./src/*.cu as lib
file(GLOB_RECURSE CORE_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cu")

if(CORE_SOURCES)
    add_library(cuda_kernels_core STATIC ${CORE_SOURCES})
    target_include_directories(cuda_kernels_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cuda_kernels_core PRIVATE CUDA::cudart cuDNN::cuDNN CUDA::cublas)
    set_optimization_flags(cuda_kernels_core)
endif()

# set output
foreach(test_source IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${test_name} PRIVATE cuda_kernels_core)

    # set arch
    set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
    set_optimization_flags(${test_name})

endforeach()