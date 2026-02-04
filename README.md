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

运行项目项目参照以下命令行：

```bash
cmake -S . -B ./build
make -C ./build
```


**2. 已实现的 CUDA 算子**

| 算子 | 头文件 | 实现版本 | 源码位置 | 测试文件 | 状态 |
|------|--------|----------|----------|----------|------|
| **Matmul** | `include/kernels/matmul.h` | cublas<br>mma<br>tiled_v1<br>tiled_v2<br>tiled_dbuf | `src/matmul/cublas.cu`<br>`src/matmul/mma.cu`<br>`src/matmul/tiled_v1.cu`<br>`src/matmul/tiled_v2.cu`<br>`src/matmul/tiled_dbuf.cu` | `tests/test_matmul.cu` | ✅ |
| **Attention** | `include/kernels/attention.h` | cublas<br>flash_v1 | `src/attention/cublas.cu`<br>`src/attention/flash_v1.cu` | `tests/test_sdpa_attn.cu` | ✅ |
| **Conv2D** | `include/kernels/conv2d.h` | implgemm<br>winograd | `src/conv2d/im2col_gemm.cu`<br>`src/conv2d/winograd.cu` | `tests/test_conv2d.cu` | ✅ |
| **Elementwise Mul** | `include/kernels/elementwise_mul.h` | vector | `src/elementwise_mul/vector.cu` | `tests/test_elementwise_mul.cu` | ✅ |
| **Softmax** | `include/kernels/softmax.h` | native | `src/softmax/native.cu` | `tests/test_softmax.cu` | ✅ |
