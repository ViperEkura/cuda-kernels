#include "kernels/softmax.h"

static constexpr int BD = 128;

__global__ void softmax_native(softmax_param_t param)
{
    const int stride = param.stride;
    const int size   = param.size;
    float*    src    = param.src;
    float*    dst    = param.dst;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    float max_val = -INFINITY;
    for (int i = 0; idx + i * stride < size; i++) {
        float val = src[idx + i * stride];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; idx + i * stride < size; i++) {
        sum_exp += expf(src[idx + i * stride] - max_val);
    }

    for (int i = 0; idx + i * stride < size; i++) {
        dst[idx + i * stride] = expf(src[idx + i * stride] - max_val) / sum_exp;
    }
}


void launch_softmax_native(softmax_param_t param)
{
    dim3 block(BD);
    dim3 grid((param.stride + BD - 1) / BD);

    softmax_native<<<grid, block>>>(param);

}