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
