### 卷积


卷积神经网络中单个输出的计算公式：

$$
Y_{i,k,x,y} = \sum_{c=1}^{C} \sum_{r=1}^{R} \sum_{s=1}^{S} D_{i,c,x+r,y+s} G_{k,c,r,s}
$$

## 公式中各符号的含义：

| 符号 | 含义 | 说明 |
|------|------|------|
| $Y_{i,k,x,y}$ | 输出特征图 | 第 $i$ 个样本，第 $k$ 个输出通道，在空间位置 $(x,y)$ 处的值 |
| $D_{i,c,x+r,y+s}$ | 输入特征图 | 第 $i$ 个样本，第 $c$ 个输入通道，在位置 $(x+r, y+s)$ 处的值 |
| $G_{k,c,r,s}$ | 卷积核权重 | 第 $k$ 个输出通道，第 $c$ 个输入通道，在位置 $(r,s)$ 处的权重值 |
| $C$ | 输入通道数 | 输入特征图的通道数量 |
| $R$ | 卷积核高度 | 卷积核在垂直方向的大小 |
| $S$ | 卷积核宽度 | 卷积核在水平方向的大小 |


**输入** : [b, c, h, w]

**权重** : [k, c, r, s]

**输出** : [b, k, Oh, Ow]



由此得到朴素版本的卷积如下



```c
void conv2dcpu(float* pin, float* pwei, float* pout, 
int n, int c, int h, int w, int k, int r, int s, int u, int v,  int p, int q)
{
    int oh = (h + 2*p - r)/u + 1;
    int ow = (w + 2*q - s)/v + 1;
    
    for(int nNum = 0; nNum < n; nNum++)
    {
        for(int kNum = 0; kNum< k; kNum++)
        {
            for(int i=0; i<oh; i++)
            {
                for(int j = 0; j< ow; j++)
                { 
                    double sum = 0.0;
                    int posh = i*u - p;
                    int posw = j*v - q;
                    for(int cNum = 0; cNum < c; cNum++)
                    {                       
                        for(int khNum = 0; khNum < r; khNum++)
                        {
                            for(int kwNum = 0; kwNum < s; kwNum++)
                            {
                                int posh_ori = posh + khNum;
                                int posw_ori = posw + kwNum;
                                if(posw_ori >= 0 && posh_ori >= 0 && posw_ori < w  && posh_ori < h)
                                {
                                    sum += (double)(pin[nNum*c*h*w + cNum*(w*h)+ posh_ori*w + posw_ori] * pwei[kNum*r*s*c + cNum*r*s + khNum*s + kwNum]);
                                }
                            }                       
                        }
                    }

                    pout[nNum*k*oh*ow + kNum*oh*ow + i*ow + j] = (float)sum;
                }
            }
        }
    }
}

```

对于并行化， 我们可以采用并行掉b,k,Oh,Ow维度， 内部只用处理 c, r s, 维度进行计算

