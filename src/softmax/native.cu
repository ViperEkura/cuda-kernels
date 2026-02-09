#include "kernels/softmax.h"
#include <stdexcept>


static constexpr int Bd = 256;

__global__ void softmax_native(softmax_param_t param)
{
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = chunk_id * param.softmax_stride * param.softmax_size;

    float max_val = -INFINITY;
    double exp_sum_val = 0;

    if (base_idx >= param.total_size) return;

    // find max
    for (int i = 0; i < param.softmax_size; i++)
    {
        int idx = base_idx + i * param.softmax_stride;
        max_val = fmaxf(max_val, param.src[idx]);
    }

    // exp
    for (int i = 0; i < param.softmax_size; i++)
    {
        int idx = base_idx + i * param.softmax_stride;
        exp_sum_val += __expf(param.src[idx] - max_val);
    }

    //reduce
    float inv_sum = 1.0f / exp_sum_val;
    for(int i = 0; i < param.softmax_size; i++)
    {
        int idx = base_idx + i * param.softmax_stride;
        param.dst[idx] = __expf(param.src[idx] - max_val) * inv_sum;
    }
}


void launch_softmax_native(softmax_param_t param)
{
    int chunks = param.total_size / (param.softmax_size * param.softmax_stride);

    dim3 block(Bd);
    dim3 grid((chunks + Bd - 1) / Bd);
    softmax_native<<<grid, block>>>(param);

}