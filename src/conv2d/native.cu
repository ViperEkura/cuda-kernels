#include "kernels/conv2d.h"

__global__ void conv2d_native(conv2d_param_t param){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    if(x >= param.Oh*param.Ow || y >= param.k || z >= param.n)return;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/param.Ow;
    int posOw = x%param.Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    float sum = 0.0;

    int inOffset = z*param.c*param.h*param.w + posh_ori*param.w + posw_ori;
    int weiOffset = y*param.c*param.r*param.s;
    int inChannelOffset = param.h*param.w;
    int weightChannelOffset = param.r*param.s;
    
    for(int i = 0; i < param.r; i++)
    {
        for(int j = 0; j < param.s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;            
            
            if(posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for(int channel = 0; channel<param.c; channel++)
                {
                    sum += (float)(param.in[inOffsetTmp + i*param.w + j] * param.weight[weiOffsetTmp + i*param.s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }               
            }
        }
    }   

    //计算输出偏移
    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow + x;
    param.out[outOffset] = sum;

}

void launch_conv2d_native(conv2d_param_t param){
    unsigned int n = param.n;
    //unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    int blockx   = (outh*outw + 15) / 16; 
    int blocky   = (k + 15)/16; 
    int blockz   = n;    
    int threadx  = 16;
    int thready  = 16;
    int threadz  = 1; 

    dim3 block(blockx, blocky, blockz);
    dim3 thread(threadx, thready, threadz);
    conv2d_native<<<block, thread>>>(param);
}
