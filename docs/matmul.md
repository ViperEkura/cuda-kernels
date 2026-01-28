## 矩阵乘法

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
            sum += lhs[M][K] * rhs[K][N];
        }
        dst[M][N] = sum;
    }
}
```

常见的方法是对M, N维度进行分块， 并且利用共享内存 尽可能一次性计算更多的K 维度分块
另外还可以通过双缓冲对读取的延迟进行掩盖