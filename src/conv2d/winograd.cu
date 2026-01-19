#ifndef WINOGRAD_CU
#define WINOGRAD_CU

#include "kernels/conv2d.h"

__device__ void winograd_4x4_3x3(float* g, float* d,  float* o){
    //g[3,3], d[6, 6], o[4, 4]
    int pos = 0, org_pos = 0;
    
    for(;org_pos < 24; org_pos += 6, pos += 4){
        o[pos] += g[0] * d[org_pos] + g[1] * d[org_pos+ 1] + g[2] * d[org_pos + 2] \
                   + g[3] * d[org_pos + 6] + g[4] * d[org_pos + 7] + g[5] * d[org_pos + 8] \
                   + g[6] * d[org_pos + 12] + g[7] * d[org_pos + 13] + g[8] * d[org_pos + 14];

        o[pos + 1] += g[0] * d[org_pos + 1] + g[1] * d[org_pos+ 2] + g[2] * d[org_pos + 3] \
                    + g[3] * d[org_pos + 7] + g[4] * d[org_pos + 8] + g[5] * d[org_pos + 9] \
                    + g[6] * d[org_pos + 13] + g[7] * d[org_pos + 14] + g[8] * d[org_pos + 15];

        o[pos + 2] += g[0] * d[org_pos + 2] + g[1] * d[org_pos+ 3] + g[2] * d[org_pos + 4] \
                    + g[3] * d[org_pos + 8] + g[4] * d[org_pos + 9] + g[5] * d[org_pos + 10] \
                    + g[6] * d[org_pos + 14] + g[7] * d[org_pos + 15] + g[8] * d[org_pos + 16];

        o[pos + 3] += g[0] * d[org_pos + 3] + g[1] * d[org_pos+ 4] + g[2] * d[org_pos + 5] \
                    + g[3] * d[org_pos + 9] + g[4] * d[org_pos + 10] + g[5] * d[org_pos + 11] \
                    + g[6] * d[org_pos + 15] + g[7] * d[org_pos + 16] + g[8] * d[org_pos + 17];
    }
}

__global__ void conv2d_winograd(conv2d_param_t param){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int posOh = x / ((param.Ow + 3) >> 2) << 2;
    int posOw = x % ((param.Ow + 3) >> 2) << 2;

    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    int inOffsetBase  = z*param.c*param.h*param.w;
    int weiOffsetBase = y*param.c*param.r*param.s;
    int inChannelStep = param.h*param.w;
    int weiChannelStep = param.r*param.s;
    
    int inOffsetTmp = inOffsetBase;
    int weiOffsetTmp = weiOffsetBase;
    __shared__ float smemin[1152][8];
    float memout[16];
    float d[36], g[9];
    memset(memout, 0, sizeof(memout));
    int c = 0;
    if (param.k >= 64){
        for(; c + 31 < param.c; c+= 32){
            for(int i=0;i<6;++i){
                for(int j=0;j<6;++j){
                    int posh_real = posh_ori + i;
                    int posw_real = posw_ori + j;
                    smemin[threadIdx.y * 36 + i * 6 + j][threadIdx.x] = (posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)? \
                        param.in[inOffsetTmp + threadIdx.y * inChannelStep  + posh_real * param.w + posw_real] : 0;
                }
            }
            __syncthreads();
            
            for(int i=0;i<32;++i){
                int im36 = i * 36;
                for(int j=0;j<9;++j) g[j] = param.weight[weiOffsetTmp + j];
                for(int j=0;j<36;++j) d[j] = smemin[im36 + j][threadIdx.x];
                winograd_4x4_3x3(g, d, memout);
                weiOffsetTmp +=  weiChannelStep;
            }
            inOffsetTmp += 32 * inChannelStep;
            __syncthreads();
        }
    }
    if(posOh >= param.Oh || posOw >= param.Ow|| y >= param.k || z >= param.n) return;
    
    for(; c < param.c; ++c){
        for(int i=0;i<6;++i){
            for(int j=0;j<6;++j){
                int posh_real = posh_ori + i;
                int posw_real = posw_ori + j;
                d[i * 6 + j] = (posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)? \
                    param.in[inOffsetTmp + posh_real * param.w + posw_real] : 0;
            }
        }
        winograd_4x4_3x3(param.weight + weiOffsetTmp, d, memout);
        inOffsetTmp += inChannelStep;
        weiOffsetTmp += weiChannelStep;
    }
    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow;
    for(int i=0;i<4;++i){
        for(int j=0;j<4;++j){
            int oh = posOh + i;
            int ow = posOw + j;
            if(oh < param.Oh && ow < param.Ow){
                param.out[outOffset + oh * param.w + ow] = memout[i * 4 + j];
            }
        }
    }
}

void launch_winograd(conv2d_param_t param){
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

    int infoH = ((outh + 3) / 4) * 4;
    int infoW = ((outw + 3) / 4) * 4;
    int blockx   = (infoH*infoW + 127) / 128; 
    int blocky   = (k + 31) / 32; 
    int blockz   = n;    
    int threadx  = 8;
    int thready  = 32;
    int threadz  = 1; 

    dim3 block(blockx, blocky, blockz);
    dim3 thread(threadx, thready, threadz);
    conv2d_winograd<<<block, thread>>>(param);
}

#endif