#include "elementwise_mul/func.h"

__global__ void elementwise_mul(elementwise_mul_param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = bx * blockDim.x + tx;
    if (idx >= param.p_size) return;

    param.A[idx] = param.A[idx] * param.alpha + param.B[idx];
}

void launch_elementwise_mul(elementwise_mul_param_t param)
{
    int thread = 32;
    int block = (param.p_size + thread - 1) / thread;
    
    elementwise_mul<<<block, thread>>>(param);
}