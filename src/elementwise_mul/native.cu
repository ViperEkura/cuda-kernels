#include "kernels/elementwise_mul.h"

static constexpr int THRED = 256;
static constexpr int TILE_SIZE = 32;

__global__ void elementwise_mul_native(elementwise_mul_param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int block_start = bx * blockDim.x * TILE_SIZE;
    int block_end = min(block_start + blockDim.x * TILE_SIZE, param.N);
    
    // block stride
    for (int offset = block_start + tx; offset < block_end; offset += blockDim.x) {
        param.dst[offset] = param.lhs[offset] * param.rhs[offset];
    }
}

void launch_elementwise_mul_native(elementwise_mul_param_t param)
{
    int seg_size = TILE_SIZE * THRED;
    int block = (param.N + seg_size - 1) / seg_size;
    elementwise_mul_native<<<block, THRED>>>(param);
}
