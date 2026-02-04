## Attention Forward

**1.公式**

$$
\begin{aligned}
p_{m, n} &= \text{softmax}\left(\sum_d q_{m, d} k_{n, d} s\right) \newline
         &= \frac{\exp\left(\sum_d  q_{m, d} k_{n, d}s\right)}{\sum_n \exp\left(\sum_d  q_{m, d} k_{n, d} s\right)} \newline
\newline
o_{m, d} &= \sum_n p_{m, n} v_{n, d} \newline
         &= \sum_n \frac{\exp\left(\sum_d q_{m, d} k_{n, d}s\right)}{\sum_n \exp\left(\sum_d q_{m, d} k_{n, d} s\right)} v_{n, d}
\end{aligned}
$$

则对于计算单个输出

$$
o = \sum_n \frac{exp(\sum_d q_{d} k_{n, d}s)}{\sum_n exp(\sum_d q_{d} k_{n, d} s)} v_{n} \newline
$$


**2.伪代码**

对于最简单的实现，在基础的q长度for循环内部还需要四个for 循环 进行实现, 大致全局内存访问次数为 2MND

```python
for m in range(M):
    score = tensor(N)
    weight = tensor(N)
    output = tensor(D) 
    # 1. 计算点积
    for n in range(N):
        dot = 0
        for d in range(D):
            dot += Q[m, d] * K[n, d]
     	scores[n] = exp(scale * dot)
    
    # 2. 计算分母（softmax归一化）
    denominator = 0
    for n in range(N):
        denominator += exp(scores[n])
    
    # 3. 归一化得到权重
    for n in range(N):
        weights[n] = exp(scores[n]) / denominator
    
    # 4. 计算输出
    for n in range(N):
    	for d in range(D):
            output[d] += weights[n] * V[n, d]
```

为了维持数值稳定性和融合计算， 可以使用online softmax 进行加速， 从而将对全局内存的访问降低到MND, 这一版本为native 版本

```python
# 完全融合版本 - 减少内存访问
for m in range(M):
    max_score = -inf
    denominator = 0
    output_row = tensor(D)
    
    # 单次遍历N，完全融合计算
    for n in range(N):
        # 1. 计算点积
        dot = 0
        for d in range(D):
            dot += Q[m, d] * K[n, d]
        
        scaled_dot = scale * dot
        
        # 2. 更新最大值（用于数值稳定）
        if scaled_dot > max_score:
            # 需要重新调整之前的计算结果
            if n > 0:
                rescale = exp(max_score - scaled_dot)
                denominator *= rescale
                for d in range(D):
                    output_row[d] *= rescale
            max_score = scaled_dot
        
        # 3. 计算当前项的贡献
        exp_val = exp(scaled_dot - max_score)
        denominator += exp_val
        
        # 4. 立即累加到输出（融合权重计算和累加）
        weight = exp_val  # 先不除以denominator，最后统一除
        for d in range(D):
            output_row[d] += weight * V[n, d]
    
    # 最后统一归一化
    for d in range(D):
        output[m, d] = output_row[d] / denominator
```

另外， 为了使用共享显存，可以对当前计算进行分块， 沿着KV 维度进行 online softmax

```python
def flash_attention(Q, K, V, M=None, eps=1e-5):
    N, d = Q.shape
    device = Q.device
    
    # 计算块大小
    B_c = math.ceil(M / (4 * d))  # K, V块的序列长度
    B_r = min(math.ceil(M / (4 * d)), d)  # Q块的序列长度
    
    # 2. 在HBM中初始化输出和中间状态
    O = torch.zeros(N, d, device=device)  # 输出矩阵
    l = torch.zeros(N, device=device)     # 分母累加器
    m = torch.full((N,), -float('inf'), device=device)  # 最大值记录器
    
    # 3. 分块
    T_r = math.ceil(N / B_r)  # Q的块数
    T_c = math.ceil(N / B_c)  # K, V的块数
    
    # 4. 外层循环：遍历K, V块
    for j in range(T_c):
        # 6. 加载K_j, V_j到SRAM
        K_start = j * B_c
        K_end = min(K_start + B_c, N)
        K_j = K[K_start:K_end, :]  # [B_c, d]
        V_j = V[K_start:K_end, :]  # [B_c, d]
        
        # 7. 内层循环：遍历Q块
        for i in range(T_r):
            Q_start = i * B_r
            Q_end = min(Q_start + B_r, N)
            
            # 8. 加载Q_i, O_i, l_i, m_i到SRAM
            Q_i = Q[Q_start:Q_end, :]  # [B_r, d]
            l_i = l[Q_start:Q_end]     # [B_r]
            m_i = m[Q_start:Q_end]     # [B_r]
            
            # 9. 在片上计算S_{ij} = Q_i @ K_j^T
            S_ij = torch.matmul(Q_i, K_j.T)  # [B_r, B_c]
            
            # 10. 计算当前块的softmax统计量
            m_ij_tilde = torch.max(S_ij, dim=1).values  # [B_r] 行最大值
            P_ij_tilde = torch.exp(S_ij - m_ij_tilde.unsqueeze(1))  # [B_r, B_c]
            l_ij_tilde = torch.sum(P_ij_tilde, dim=1)  # [B_r] 行和
            
            # 11. 更新全局softmax统计量
            m_i_new = torch.maximum(m_i, m_ij_tilde)  # [B_r]
            
            # 计算缩放因子
            scale_old = torch.exp(m_i - m_i_new)  # [B_r]
            scale_new = torch.exp(m_ij_tilde - m_i_new)  # [B_r]
            l_i_new = scale_old * l_i + scale_new * l_ij_tilde  # [B_r]
            
            # 12. 更新输出
            PV = P_ij_tilde @ V_j
            O[Q_start:Q_end, :] = O[Q_start:Q_end, :] * scale_old + PV * scale_new


            l[Q_start:Q_end] = l_i_new
            m[Q_start:Q_end] = m_i_new
    

    O =  O / (l + eps)
    
    return O

```
