#ifndef ELEMENTWISE_MUL_CU
#define ELEMENTWISE_MUL_CU

#include "kernels/elementwise_mul.h"

static constexpr int TILE_SIZE = 32;

__global__ void elementwise_mul_verify(elementwise_mul_param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int base_idx = (bx * blockDim.x + tx) * TILE_SIZE;
    
    int residual_idx = (param.p_size - base_idx) > TILE_SIZE ? TILE_SIZE: (param.p_size - base_idx);
    for (int i = 0; i < residual_idx; i++) {
        int idx = base_idx + i;
        param.A[idx] = param.A[idx] * param.alpha + param.B[idx];
    }
}

void launch_elementwise_mul_verify(elementwise_mul_param_t param)
{
    int thread = 32;
    int seg_size = TILE_SIZE * 32;
    int block = (param.p_size + seg_size - 1) / seg_size;
    
    elementwise_mul_verify<<<block, thread>>>(param);
}

#endif