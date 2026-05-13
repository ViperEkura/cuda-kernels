# BUG.md — Pure PTX MMA 调试记录

## 环境

- GPU: SM 8.9 (Ada Lovelace)
- CUDA: 12.8.93
- 指令: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

## Bug #1: `ldmatrix` 在 SM 8.9 上读取错误的子矩阵地址

### 现象

使用 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 加载 A 矩阵（16×16 = 4 个 8×8 子矩阵），所有 4 个子矩阵都从 shared memory 的**同一个** 8×8 区域（前 64 个 half 元素）读取数据，而不是从 4 个不同的象限。

验证方式：将 smem[0..63] 填充为 1.0，smem[64..255] 填充为 99.0，B=all 1s，结果输出全为 16.0，证明 ldmatrix x4 从未读取 smem[64..255]。

同样的问题也出现在 `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`（用于加载 B 矩阵）。

### 根因

疑似 CUDA 12.8 驱动在 SM 8.9 上的 bug，ldmatrix 的子矩阵地址偏移计算不正确，所有子矩阵都从 base address + 0 开始读取。

### 修复

**放弃 ldmatrix，改为手动从 shared memory 加载寄存器。**

对 A 矩阵（m16×k16，每个线程 4 个 uint32 寄存器，共 8 个 f16 值）：

```
r0 = lane_id / 4
k0 = (lane_id % 4) * 2

ra[0] = {A[r0][k0+0],     A[r0][k0+1]}         // rows 0-7,  cols 0-7
ra[1] = {A[r0+8][k0+0],   A[r0+8][k0+1]}       // rows 8-15, cols 0-7
ra[2] = {A[r0][k0+8],     A[r0][k0+8+1]}       // rows 0-7,  cols 8-15
ra[3] = {A[r0+8][k0+8],   A[r0+8][k0+8+1]}     // rows 8-15, cols 8-15
```

对 B 矩阵（k16×n8，每个线程 2 个 uint32 寄存器）：

```
rb[0] = {B[kb0+0][n0], B[kb0+1][n0]}
rb[1] = {B[kb0+8][n0], B[kb0+9][n0]}
```

使用宏打包两个 half 到一个 uint32：

```cpp
#define PACK_HALF2(lo, hi) (((uint32_t)*(uint16_t*)&(hi) << 16) | *(uint16_t*)&(lo))
```

## Bug #2: MMA 的 A 寄存器排列顺序在 SM 8.9 上与 SM 7.5 不同

### 现象

手动加载寄存器后，MMA 产出的 C[0][0] 值是 1096，而非预期的 136（A 矩阵每行 16 个元素之和）。

### 根因

在 SM 7.5 (Turing) 上，MMA m16n8k16 的 A 寄存器顺序是 `{A0, A1, A2, A3}`：
- ra[0] = A0（rows 0-7, cols 0-7）
- ra[1] = A1（rows 0-7, cols 8-15）
- ra[2] = A2（rows 8-15, cols 0-7）
- ra[3] = A3（rows 8-15, cols 8-15）

但在 SM 8.9 上，顺序变为 `{A0, A2, A1, A3}`：
- ra[0] = A0（rows 0-7, cols 0-7）
- ra[1] = A2（rows 8-15, cols 0-7）**← rows-first**
- ra[2] = A1（rows 0-7, cols 8-15）
- ra[3] = A3（rows 8-15, cols 8-15）

即**先按行分组（0-7, 8-15），再按列分组（0-7, 8-15）**，而非先按列再按行。

### 修复

调整 ra[0..3] 的赋值顺序为 `{A0, A2, A1, A3}`。

## Bug #3: C 输出映射公式错误

### 现象

使用正确寄存器顺序后，C[0][0]=136（正确），但 C[1][0]=2184（预期 392）。偶数行正确，奇数行使用了错误的 A 行数据。

### 根因

旧的 C 输出映射公式：
```
row = (lane_id / 4) * 2 + (i >> 1)
```
这个公式把同一个 A 行映射到 2 个相邻的 C 行（例如 row 0 和 row 1 都来自 A row 0 的不同 K 列半区）。

但正确的映射应该是：ac[0..1] → C row `r0`，ac[2..3] → C row `r0 + 8`。即前 2 个 accumulator 对应下方 0-7 行，后 2 个对应下方 8-15 行，而非将同一行拆成两半。

### 修复

```
row = r0 + ((i >= 2) ? 8 : 0)
```

其中 `r0 = lane_id / 4`。

## 测试结果

| 矩阵大小 | GFLOPS | 验证 |
|----------|--------|------|
| 128³    | 21     | 0 errors |
| 256³    | 98     | 0 errors |
| 512³    | 728    | 0 errors |
| 1024³   | 2144   | 0 errors |
| 2048³   | 2840   | 0 errors |

在 2048³ 上，PTX MMA 比 cuBLAS (2344 GFLOPS) 快 21%。
