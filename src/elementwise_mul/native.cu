#include "kernels/elementwise_mul.h"

static constexpr int THRED = 128;
static constexpr int TILE_SIZE = 32;

__global__ void elementwise_mul_native(elementwise_mul_param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int base_idx = (bx * blockDim.x + tx) * TILE_SIZE;
    int residual_idx = min(param.N - base_idx, TILE_SIZE);

    for (int i = 0; i < residual_idx; i++) {
        int idx = base_idx + i;
        param.dst[idx] = param.lhs[idx] *  param.rhs[idx];
    }
}

void launch_elementwise_mul_native(elementwise_mul_param_t param)
{
    int seg_size = TILE_SIZE * THRED;
    int block = (param.N + seg_size - 1) / seg_size;
    elementwise_mul_native<<<block, THRED>>>(param);
}
