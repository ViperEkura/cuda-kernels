## cuda-kernels

**1.项目结构**

项目文件结构如下：

```bash
.
├── CMakeLists.txt
├── README.md
├── docs
│   └── ...
├── include
│   ├── common.h
│   └── kernels
│       ├── operator
│       ├── ...
│       └── ...
├── src
│   ├── operator
│   │   ├── operator_impl.cu
│   │   ├── ...
│   │   └── ...
│   └── ...
└── tests
    ├── test_operator.cu
    ├── ...
    └── ...
```

构建项目项目参照以下命令行：

```bash
cmake -S . -B ./build
make -C ./build -j4
```

在构建项目之后可以进行对应算子的性能分析：

```bash
python scripts/operator_bench.py
```

对应性能分析结果存储在 performance 文件夹目录下

**2. 已实现的 CUDA 算子**


| 算子 | 头文件 | 实现版本 | 核心优化 | 备注 |
|:-----|:-------|:---------|:---------|:-----|
| **Matmul** | `include/kernels/matmul.h` | | | |
| | | native | 基础实现 | / |
| | | cublas | cuBLAS封装 | 对比用 |
| | | mma | Tensor Core + WMMA API | FP16混合精度 |
| | | tiled_v1 | 基础分块 | 减少HBM访问 |
| | | tiled_v2 | Bank Conflict优化 | 增加Swizzling |
| | | tiled_v3 | Bank Conflict优化 + Vectorize | 增加float4 加载 |
| | | tiled_dbuf | Bank Conflict优化 + Double Buffer | 增加双缓冲|
| **Attention** | `include/kernels/attention.h` | | | |
| | | native | 基础实现 | / |
| | | cublas | 基于cuBLAS实现 | 内存密集 |
| | | flash_v1 | 增加分块 | 减少HBM访问 |
| | | flash_v2 | 增加线程计算量  | 优化warp调度 |
| **Conv2D** | `include/kernels/conv2d.h` | | | |
| | | native | 基础实现 | / |
| | | implgemm | Implicit GEMM | 转化为矩阵乘 |
| | | winograd | Winograd算法 | 适用于3x3卷积 |
| **Elementwise Mul** | `include/kernels/elementwise_mul.h` | | | |
| | | native | 基础实现 | / |
| | | vector | 向量化加载 | float4优化 |
| **Softmax** | `include/kernels/softmax.h` | | | |
| | | native | 基础实现 | / |
| | | smem | 使用共享显存 | 减少HBM访问 |