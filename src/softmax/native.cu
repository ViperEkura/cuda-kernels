#include "kernels/softmax.h"
#include <stdexcept>


static constexpr int Bd = 256;

__global__ void softmax_native(softmax_param_t param)
{
    int chunks = param.outer_size * param.inner_size;
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (chunk_id > chunks) return;

    int outer_id = chunk_id / param.inner_size;
    int inner_id = chunk_id % param.inner_size;
    int offset = outer_id * param.inner_size * param.softmax_size + inner_id;

    float max_val = -INFINITY;
    float exp_sum_val = 0;

    // find max
    for (int i = 0; i < param.softmax_size; i++)
    {
        max_val = max(max_val, param.src[offset + i * param.inner_size]);
    }

    // calcu exp
    for (int i = 0; i < param.softmax_size; i++)
    {
        exp_sum_val += exp(param.src[offset + i * param.inner_size] - max_val);
    }
    
    // write back
    float inv_exp_sum_val = 1 / exp_sum_val;
    for (int i = 0; i < param.softmax_size; i++)
    {
        float res = exp(param.src[offset + i * param.inner_size] - max_val) * inv_exp_sum_val;
        param.dst[offset + i * param.inner_size] = res;
    }
}


void launch_softmax_native(softmax_param_t param)
{
    int threads = Bd;
    int chunks = param.outer_size * param.inner_size;
    int blocks = (chunks + threads - 1) / threads;
    softmax_native<<<blocks, threads>>>(param);

}