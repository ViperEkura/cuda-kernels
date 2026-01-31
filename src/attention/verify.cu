#include "kernels/attention.h"

static constexpr int BB = 4;
static constexpr int BQ = 4;
static constexpr int BD = 64;
static constexpr int TD = 4;


__global__ void sdqa_attention_fwd_native(attention_param_t param)
{
    const int B = param.batch;
    const int D = param.dim;
    const int L_q = param.len_q;
    const int L_kv = param.len_kv;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const int b_id  = bz * blockDim.z + tz;
    const int lq_id = by * blockDim.y + ty;
    const int d_id  = bx * blockDim.x + tx;
    const int q_offset   = b_id * L_q * D + lq_id * D;
    const int o_offset   = b_id * L_q * D + lq_id * D + d_id;

    if (b_id >= B || lq_id >= L_q || d_id >= D) return;
    
    float output_cache[TD];
    float max_score = -INFINITY;
    float reduce_scale = 0;

    for(int td = 0; td < TD; td++)
    {
        output_cache[td] = 0;
    }

    // for b in range(B)
    // for l_q in range(L_q)
    for (int l_kv = 0; l_kv < L_kv; l_kv++)
    {
        float qk_scaled = 0;
        const int kv_offset = b_id * L_kv * D + l_kv * D;

        // dot product can't be devided
        for(int d = 0; d < D; d++)
        {
            qk_scaled += param.q_ptr[q_offset + d] * param.k_ptr[kv_offset + d];
        }
        
        qk_scaled *= param.scale;

        if (qk_scaled > max_score)
        {
            float rescale = exp(max_score - qk_scaled);
            max_score = qk_scaled;
            reduce_scale *= rescale;
            // split across the d dimension
            for(int td = 0; td < TD; td++)
            {
                output_cache[td] *= rescale;
            }
        }

        float exp_val = exp(qk_scaled - max_score);
        reduce_scale += exp_val;
        // split across the d dimension
        for(int td = 0; td < TD && td + d_id < D; td++)
        {
            output_cache[td] += exp_val * param.v_ptr[kv_offset + d_id + td];
        }
    }
    
    //scale at end
    for(int td = 0; td < TD && td + d_id < D; td++)
    {
        param.o_ptr[o_offset + td] = output_cache[td] / (reduce_scale + param.eps);
    }
}


void launch_sdqa_attention_fwd_native(attention_param_t param)
{
    int b = param.batch;
    int lq = param.len_q;
    int d = param.dim;
    dim3 block(BD / TD, BQ, BB);
    dim3 grid((d + BD - 1) / BD ,(lq + BQ - 1) / BQ, (b + BB - 1) / BB);
    sdqa_attention_fwd_native<<<grid, block>>>(param);

}