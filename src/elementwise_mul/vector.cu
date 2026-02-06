#include "kernels/elementwise_mul.h"

static constexpr int TILE_SIZE = 8;

__global__ void elementwise_mul_vector(elementwise_mul_param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int base_idx = (bx * blockDim.x + tx) * TILE_SIZE;
    int residual_idx = min(param.N - base_idx, TILE_SIZE);

    int i = 0;
    for(; i + 3 < residual_idx; i += 4) {
        int idx = base_idx + i;
        float4 lhs = __ldg(reinterpret_cast<float4*>(param.lhs + idx));
        float4 rhs = __ldg(reinterpret_cast<float4*>(param.rhs + idx));
        float4 dst = make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
        *reinterpret_cast<float4*>(param.dst + idx) = dst;
    }

    for (; i < residual_idx; i++) {
        int idx = base_idx + i;
        param.dst[idx] = param.lhs[idx] *  param.rhs[idx];
    }
}

void launch_elementwise_mul_vector(elementwise_mul_param_t param)
{
    int thread = 32;
    int seg_size = TILE_SIZE * thread;
    int block = (param.N + seg_size - 1) / seg_size;
    
    elementwise_mul_vector<<<block, thread>>>(param);
}
