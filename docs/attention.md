## Attention Forward

**1.公式**

注意力机制是现代深度学习的核心组件之一，特别是Transformer架构的成功，使注意力机制成为自然语言处理、计算机视觉和多模态领域的基础算子。然而，标准的注意力计算存在显著的计算瓶颈和内存访问问题，限制了模型规模和训练效率。

常见的缩放点积实现方式如下：

$$
\begin{aligned}
S_{ij} &= \frac{Q_iK_j}{\sqrt{d_k}} \newline
P_{ij} &= \frac{\text{exp}(S_{ij})}{\sum_{n=1}^{N}\text{exp}(S_{ij})} \newline
O_{id} &= PV \newline
where &\quad Q \in \mathbb{R}^{M \times d} \quad K, V \in \mathbb{R}^{N \times d}
\end{aligned}
$$

在做算子优化的时候， 需要分清楚这个算子是compute bound 还是 memory bound， 另外还需要考虑到GPU 的分级内存结构， 我们通常使用算数强度进行分析，其中算数强度（Arithmetic Intensity）是指在一个计算过程中，每从内存中搬运1字节数据，所执行的浮点运算次数，对于原始实现的数学计算强度分析如下：

对于计算强度：

1. $S = \frac{QK^T}{\sqrt{d_k}}$ : $2MNd$ FLOPS
2. $P = \text{softmax}(S)$ : $5MN$ FLOPS
3. $O = PV$ : $2MNd$ FLOPS

总计算强度为 $4MNd + 5MN$ FLOPs

对于内存访问数量, 假设 FP32 精度：

1. 读取$Q, K$: $4 \times (Md + Nd)$
2. 写回$S$: $4 \times MN$
3. 读取$S$: $4 \times MN$
4. 写回$P$: $4 \times MN$
5. 读取$P, V$: $4 \times (MN + Nd)$
6. 写回$O$: $4 \times Md$

总访问量为 $4 \times (2Md + 2Nd + 4MN)$

求得算数强度的表达式为： 

$AI = \frac{FLOPs}{Bytes} = \frac{4MNd + 5MN}{4 \times (2Md + 2Nd + 4MN)}$, 

我们选用一个比较常见的训练参数 $M = N = 2048, d = 64$ 
计算得到$AI = \frac{1103929344}{ 69206016} = 15.95 \text{FLOPs}/\text{Byte}$

而对比之下， 对于矩阵乘法而言， 当三个个维度相同时并使用FP32精度， 其计算强度为 $\frac{M}{6}$, 以 $M = 1024$ 为例
计算得到 $AI = \frac{1024}{6} = 170.67 \text{FLOPs}/\text{Byte}$, 相对而言计算强度更大。

所以就数学上分析而言，attention 算子是属于memory bound 的一类， 需要对访存进行优化并且高效地存储中间值， 通过引入 OnlineSoftmax 机制， 我们可以高效地解决这一问题。传统的Softmax 计算分为三步，分别是求的指数最大值，计算指数和， 归一化， 其中不管如何都会有两次读取输入参数，并且从HBM中读取而不是从SRAM中读取。
Online softmax 是分块计算指数和并且动态更新输出，最后除以迭代后的指数和， 从而只用读取一次HBM， 节省访问。


由此分析得到 flash attention 融合版本：

$$
\begin{aligned}
&\textbf{Input: } \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}, \text{ block sizes } B_c, B_r \newline
&\textbf{for } i = 1 \text{ to } T_r = \lceil N / B_r \rceil \textbf{ do} \newline
&\quad \text{Load } \mathbf{Q}_i \text{ into SRAM} \newline
&\quad \mathbf{O}_i \gets \mathbf{0},\ \ell_i \gets 0,\ m_i \gets -\infty \newline
&\quad \textbf{for } j = 1 \text{ to } T_c = \lceil N / B_c \rceil \textbf{ do} \newline
&\quad\quad \text{Load } \mathbf{K}_j, \mathbf{V}_j \newline
&\quad\quad \mathbf{S} \gets \mathbf{Q}_i \mathbf{K}_j^\top \newline
&\quad\quad m_i^{\text{new}} \gets \max(m_i, \text{rowmax}(\mathbf{S})) \newline
&\quad\quad \tilde{\mathbf{P}} \gets \exp(\mathbf{S} - m_i^{\text{new}}) \newline
&\quad\quad \ell_i \gets e^{m_i - m_i^{\text{new}}} \ell_i + \text{rowsum}(\tilde{\mathbf{P}}) \newline
&\quad\quad \mathbf{O}_i \gets e^{m_i - m_i^{\text{new}}} \mathbf{O}_i + \tilde{\mathbf{P}} \mathbf{V}_j \newline
&\quad\quad m_i \gets m_i^{\text{new}} \newline
&\quad \textbf{end for} \newline
&\quad \mathbf{O}_i \gets \mathbf{O}_i / \ell_i \newline
&\quad L_i \gets m_i + \log(\ell_i) \newline
&\textbf{end for}
\end{aligned}
$$

**2. 伪代码**

```python
# Q: [M, d]               # Query
# K: [N, d]               # Key
# V: [N, d]               # Value
# scale = 1.0 / sqrt(d)   # scale factor

Br = block_size_q
Bc = block_size_kv
Tr = ceil(M / Br)
Tc = ceil(N / Bc)

for q_block_idx in range(Tr):
    m_prev = tensor((Br,), fill=-inf)
    l_prev = tensor((Br,), fill=0.0)
    O_acc  = tensor((Br, d), fill=0.0)

    for kv_block_idx in range(Tc):
        q_start = q_block_idx * Br
        k_start = kv_block_idx * Bc

        Q_tile = Q[batch_idx, head_idx, q_start:q_start+Br, :]   # (Br, d)
        K_tile = K[batch_idx, head_idx, k_start:k_start+Bc, :]   # (Bc, d)
        V_tile = V[batch_idx, head_idx, k_start:k_start+Bc, :]   # (Bc, d)

        S_tile = scale * (Q_tile @ K_tile.T)  # (Br, Bc)

        m_j = max(S_tile, dim=-1)             # (Br,)
        P_j = exp(S_tile - m_j[:, None])      # (Br, Bc)
        l_j = sum(P_j, dim=1)                 # (Br,)

        m_new = maximum(m_prev, m_j)          # (Br,)
        l_new = exp(m_prev - m_new) * l_prev + exp(m_j - m_new) * l_j  # (Br,)

        scale_old = exp(m_prev - m_new)[:, None]  # (Br, 1)
        scale_new = exp(m_j - m_new)[:, None]     # (Br, 1)
        O_acc = scale_old * O_acc + scale_new * (P_j @ V_tile)  # (Br, d)

        m_prev = m_new
        l_prev = l_new


        O_block = O_acc / l_prev[:, None]  # (Br, d)
        O[batch_idx, head_idx, q_start:q_start+Br, :] = O_block
```

