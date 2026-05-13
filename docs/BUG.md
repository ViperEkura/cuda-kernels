# BUG.md — Pure PTX MMA 调试记录

## 环境

- GPU: SM 8.9 (Ada Lovelace)
- CUDA: 12.8.93
- 指令: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

## Bug #1: `ldmatrix` 用法错误 → 已修正（非驱动 bug）

### 现象

最初使用 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 加载 A 矩阵时，所有 32 个线程传入**同一个基地址**，导致所有 4 个子矩阵都从 base address + 0 的同一个 row 读取数据。同样的问题出现在 B 矩阵的 x2.trans 上。

### 根因

**不是驱动 bug**，而是对 PTX ISA 文档的理解错误。查阅 PTX ISA 9.2 文档 §9.7.14.5.15 后确认，ldmatrix 的三个关键要求：

1. **Blocked 8×8 smem 布局**：ldmatrix 内部用 `addr+64half`、`addr+128half` 计算子矩阵偏移，这些偏移要求 smem 以 4 个连续 8×8 块（blocked layout）组织，而非标准行优先布局。

2. **每线程独立行地址**：32 线程分为 4 组（threads 0-7 → Q0, 8-15 → Q2, 16-23 → Q1, 24-31 → Q3），**每个线程提供自己所在行的地址**，由 `lane_id / 4` 决定行号和 `lane_id >> 3` 决定组号。

3. **B `.trans` 用 `[K][N]` 布局**：对于 x2.trans，块内必须按 `lk * 8 + n`（即 `[K-row][N-col]`）存储，使 ldmatrix 的 stride-8 读取遍历 K-rows，transpose 后得到同一 N-col 的多行 K 值。

### 修复

移除手动 `PACK_HALF2` 加载，改用正确配置的 ldmatrix（见 `src/matmul/mma.cu`）。

## Bug #2: MMA 的 A 寄存器排列顺序在 SM 8.9 上与 SM 7.5 不同

### 现象

手动加载寄存器后，MMA 产出的 C[0][0] 值是 1096，而非预期的 136（A 矩阵每行 16 个元素之和）。

### 根因

在 SM 7.5 (Turing) 上，MMA m16n8k16 的 A 寄存器顺序是 `{A0, A1, A2, A3}`：
- ra[0] = A0（rows 0-7, cols 0-7）
- ra[1] = A1（rows 0-7, cols 8-15）
- ra[2] = A2（rows 8-15, cols 0-7）
- ra[3] = A3（rows 8-15, cols 8-15）

但在 SM 8.x 上，顺序变为 `{A0, A2, A1, A3}`：
- ra[0] = A0（rows 0-7, cols 0-7）
- ra[1] = A2（rows 8-15, cols 0-7）**← rows-first**
- ra[2] = A1（rows 0-7, cols 8-15）
- ra[3] = A3（rows 8-15, cols 8-15）

即**先按行分组（0-7, 8-15），再按列分组（0-7, 8-15）**，而非先按列再按行。

ldmatrix 在 SM 8.x 上自动产生正确的 rows-first 寄存器顺序，无需手动调整。

### 修复

调整 ra[0..3] 的赋值顺序为 `{A0, A2, A1, A3}`。ldmatrix 自动匹配此顺序。

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

| 矩阵大小 | PACK_HALF2 GFLOPS | ldmatrix GFLOPS | 验证 |
|----------|-------------------|-----------------|------|
| 128³    | 23                | 30              | 0 errors |
| 256³    | 174               | 157             | 0 errors |
| 512³    | 993               | 1004            | 0 errors |
| 1024³   | 2289              | 2547            | 0 errors |
