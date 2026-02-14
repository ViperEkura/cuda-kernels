#include "kernels/elementwise_mul.h"

static constexpr int THRED = 256;
static constexpr int TILE_SIZE = 32;

__global__ void elementwise_mul_vector(elementwise_mul_param_t param)
{
   int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int block_start = bx * blockDim.x * TILE_SIZE;
    int block_end = min(block_start + blockDim.x * TILE_SIZE, param.N);
    

    constexpr int PREFETCH = 2;
    float4 lhs_buf[PREFETCH], rhs_buf[PREFETCH], dst_buf[PREFETCH];

    int base;
    for (base = block_start + tx * 4; base + 3 + (PREFETCH-1) * blockDim.x * 4 < block_end; base += blockDim.x * 4) {
        
        #pragma unroll
        for (int p = 0; p < PREFETCH; p++) {
            int prefetch_base = base + p * blockDim.x * 4;
            lhs_buf[p] = __ldg(reinterpret_cast<const float4*>(param.lhs + prefetch_base));
            rhs_buf[p] = __ldg(reinterpret_cast<const float4*>(param.rhs + prefetch_base));
        }
        
        #pragma unroll
        for (int p = 0; p < PREFETCH; p++) {
            dst_buf[p].x = lhs_buf[p].x * rhs_buf[p].x;
            dst_buf[p].y = lhs_buf[p].y * rhs_buf[p].y;
            dst_buf[p].z = lhs_buf[p].z * rhs_buf[p].z;
            dst_buf[p].w = lhs_buf[p].w * rhs_buf[p].w;
            
            int store_base = base + p * blockDim.x * 4;
            *reinterpret_cast<float4*>(param.dst + store_base) = dst_buf[p];
        }
    }
    

    for (; base + 3 < block_end; base += blockDim.x * 4) {
        float4 lhs = __ldg(reinterpret_cast<const float4*>(param.lhs + base));
        float4 rhs = __ldg(reinterpret_cast<const float4*>(param.rhs + base));
        
        float4 dst;
        dst.x = lhs.x * rhs.x;
        dst.y = lhs.y * rhs.y;
        dst.z = lhs.z * rhs.z;
        dst.w = lhs.w * rhs.w;
        
        *reinterpret_cast<float4*>(param.dst + base) = dst;
    }
    
    int remaining = block_end - base;
    if (remaining > 0) {
        for (int i = 0; i < remaining; i++) {
            param.dst[base + i] = param.lhs[base + i] * param.rhs[base + i];
        }
    }
}

void launch_elementwise_mul_vector(elementwise_mul_param_t param)
{
    int seg_size = TILE_SIZE * THRED;
    int block = (param.N + seg_size - 1) / seg_size;
    
    elementwise_mul_vector<<<block, THRED>>>(param);
}
