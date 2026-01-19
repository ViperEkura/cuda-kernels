cmake_minimum_required(VERSION 3.18)
project(CUDA_KERNELS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# 设置输出目录为 bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ====== 1. 编译核心库（可选）======
# 如果你有 src/ 中的公共 kernel 实现，可以先构建成库
file(GLOB_RECURSE CORE_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
if(CORE_SOURCES)
    add_library(cuda_kernels_core STATIC ${CORE_SOURCES})
    target_include_directories(cuda_kernels_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cuda_kernels_core PRIVATE CUDA::cudart)
endif()

# ====== 2. 自动为 tests/ 下每个 .cu 创建可执行文件 ======
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cu")

foreach(test_source IN LISTS TEST_SOURCES)
    # 获取文件名（不含路径和扩展名），如 "test_conv2d"
    get_filename_component(test_name ${test_source} NAME_WE)

    # 创建可执行文件
    add_executable(${test_name} ${test_source})

    # 链接核心库（如果有）
    if(TARGET cuda_kernels_core)
        target_link_libraries(${test_name} PRIVATE cuda_kernels_core)
    endif()

    # 添加头文件路径
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)

    # 链接 CUDA 运行时
    target_link_libraries(${test_name} PRIVATE CUDA::cudart)

    # 设置 GPU 架构等属性
    set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()