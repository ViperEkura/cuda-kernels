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
cmake -S . -B ./build -G "Unix Makefiles"
cd build/
make
```
