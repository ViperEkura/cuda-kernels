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

| 算子              | 头文件                          | 实现版本                                                                 | 测试文件                     |
|-------------------|----------------------------------|--------------------------------------------------------------------------|------------------------------|
| Matmul            | `include/kernels/matmul.h`      | cublas<br>mma<br>tiled_v1<br>tiled_v2<br>tiled_dbuf<br>native| `tests/test_matmul.cu`     |
| Attention         | `include/kernels/attention.h`   | cublas<br>flash_v1<br>flash_v2<br>native                 | `tests/test_sdpa_attn.cu`  |
| Conv2D            | `include/kernels/conv2d.h`      | implgemm<br>winograd<br>native                      | `tests/test_conv2d.cu`          |
| Elementwise Mul   | `include/kernels/elementwise_mul.h` | native<br>vector                                | `tests/test_elementwise_mul.cu` |
| Softmax           | `include/kernels/softmax.h`     | native                                              | `tests/test_softmax.cu`         |

