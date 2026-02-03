cmake_minimum_required(VERSION 3.18)
project(CUDA_KERNELS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set arch
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

macro(set_optimization_flags TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3,-v>
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
    target_link_libraries(cuda_kernels_core PRIVATE CUDA::cudart CUDA::cublas)
    set_optimization_flags(cuda_kernels_core)
endif()

# set output
foreach(test_source IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${test_name} PRIVATE cuda_kernels_core)
    set_optimization_flags(${test_name})
endforeach()