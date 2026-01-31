cmake_minimum_required(VERSION 3.18)
project(CUDA_KERNELS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CUDA_OPT_FLAGS "-O3 --use_fast_math -Xptxas -O3,-v")
set(CXX_OPT_FLAGS "-O3 -march=native -mtune=native")

# output to ./bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set ./src/*.cu as lib
file(GLOB_RECURSE CORE_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cu")

if(CORE_SOURCES)
    add_library(cuda_kernels_core STATIC ${CORE_SOURCES})
    target_include_directories(cuda_kernels_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cuda_kernels_core PRIVATE CUDA::cudart)
endif()

# set output
foreach(test_source IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    if(TARGET cuda_kernels_core)
        target_link_libraries(${test_name} PRIVATE cuda_kernels_core)
    endif()

    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cuda_kernels_core PRIVATE CUDA::cudart CUDA::cublas)

    # set arch
    set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()