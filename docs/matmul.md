# 矩阵乘法

## 基础实现

对于矩阵乘法, 常见的单核cpu实现代码如下：

```cpp
int dst[M][N], lhs[M][K], rhs[K][N];

for (int m = 0; m < M; m++)
{
    for(int n = 0; n < N; n++)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            sum += lhs[m][k] * rhs[k][n];
        }
        dst[m][n] = sum;
    }
}
```

常见的方法是对M, N维度进行分块， 并且利用共享内存尽可能一次性计算更多的K维度分块。另外还可以通过双缓冲对读取的延迟进行掩盖。

## 变体

| 文件 | 方法 | 说明 |
|------|------|------|
| `native.cu` | Naive | 无优化，16×16 线程块 |
| `tiled_v1.cu` | Tiled v1 | 共享内存分块 (16×16)，`OFFSET=1` 避免 bank conflict |
| `tiled_v2.cu` | Tiled v2 | 更大 tile (128×128)，swizzle bank 映射 |
| `tiled_v3.cu` | Tiled v3 | float4 向量化加载 (`__ldg`)，swizzle |
| `wmma.cu` | WMMA | `nvcuda::wmma` API，half 精度，SM 7.0+ |
| `mma.cu` | PTX MMA | PTX 内联汇编 `mma.sync.aligned.m16n8k16`，SM 8.0+ |
| `cublas.cu` | cuBLAS | 参考实现，用于验证基准 |

## PTX MMA 实现 (`mma.cu`)

### 架构

- 线程块: 128×128 输出 tile，16 warps (4×4)
- 每个 warp: 32×32 输出，2×4 MMA tiles
- MMA tile: m16n8k16，4 个 8×8 子矩阵 (Q0/Q1/Q2/Q3)
- 数据: fp32 输入 → fp16 shared memory → fp32 累积

### ldmatrix 加载

使用 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 加载 A，`ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` 加载 B。

**关键要求**（来自 PTX ISA 9.2 §9.7.14.5.15）：

1. **Blocked 8×8 布局**：smem 必须存为 4 个连续 8×8 块（每块 64 half），而非标准行优先。ldmatrix 内部用 `addr + 64/128/192 half` 偏移定位子矩阵。

2. **每线程独立行地址**：32 线程分为 4 组，每组 8 线程为该组的 8×8 矩阵提供行地址。`lane_id >> 3` 确定组，`lane_id & 7` 确定行。不是所有线程共用同一地址。

3. **B `.trans` 用 `[K][N]` 布局**：块内按 `lk * 8 + n` 存储（K 为行方向），ldmatrix stride-8 读取遍历 K-rows，transpose 后得到同一 N-col 的多行 K 值。

### 寄存器顺序

SM 8.x 上 MMA 期望 A 寄存器顺序为 `{Q0, Q2, Q1, Q3}`（rows-first），ldmatrix 自动匹配。C 输出映射公式：

```
row = lane_id / 4 + ((i >= 2) ? 8 : 0)
col = (lane_id % 4) * 2 + (i & 1)
```

### 性能

| 矩阵大小 | GFLOPS |
|----------|--------|
| 128³    | 30     |
| 256³    | 157    |
| 512³    | 1004   |
| 1024³   | 2547   |

CUDA 12.8 / SM 8.9 上测得。
