## cuda-kernels

用于记录cuda 学习， 项目文件结构如下：

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

## 已实现的 CUDA 算子

| 算子 | 头文件 | 实现版本 | 测试文件 | 状态 |
|------|--------|----------|----------|------|
| **Attention** | `include/kernels/attention.h` |  cublas<br> flash_v1 | `tests/test_sdpa_attn.cu` | ✅ |
| **Conv2D** | `include/kernels/conv2d.h` | implgemm<br> winograd | `tests/test_conv2d.cu` | ✅ |
| **Elementwise Mul** | `include/kernels/elementwise_mul.h` | vector | `tests/test_elementwise_mul.cu` | ✅ |
| **Matmul** | `include/kernels/matmul.h` |  cublas<br> tiled_dbuf<br> tiled_v1<br> tiled_v2 | `tests/test_matmul.cu` | ✅ |
| **Softmax** | `include/kernels/softmax.h` | native | `tests/test_softmax.cu` | ✅ |