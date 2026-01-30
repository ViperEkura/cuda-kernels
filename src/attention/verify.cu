#include "kernels/attention.h"

constexpr int B = 32;
constexpr int BQ = 32;


__global__ void sdqa_attention_fwd_native(attention_param_t param)
{
    const int B = param.batch;
    const int D = param.dim;
    const int L_q = param.len_q;
    const int L_kv = param.len_kv;

    const int by = blockIdx.y, bx = blockIdx.x;
    const int ty = threadIdx.y, tx = threadIdx.x;

    const int b_id =  bx * blockDim.x + tx;
    const int lq_id = by * blockDim.y + ty;
    const int qo_offset = b_id * L_q * D + lq_id * D;

    if (b_id >= B || lq_id >= L_q) return;

    for (int lq = 0; lq < BQ; lq++)
    {
        float max_qk_dot = -INFINITY;

        // find max max_qk_dot of on [b, l_q, l_kv]
        // O(l_kv * d)
        for(int l_kv = 0; l_kv < L_kv; l_kv++)
        {
            float qk_dot = 0;
            int kv_offset = b_id * L_kv * D + l_kv * D;
            for(int d = 0; d < D; d++)
            {
                qk_dot += param.q_ptr[qo_offset + d] * param.k_ptr[kv_offset + d];
            }
            max_qk_dot = max(qk_dot * param.scale, max_qk_dot);
        }
        
        // get exp_sum on dim [b, l_q, l_kv]
        // O(l_kv * d)
        float exp_sum = 0;
        for (int l_kv = 0; l_kv < L_kv; l_kv++)
        {
            int k_offset = b_id * L_kv * D + l_kv * D;
            float qk_dot = 0.0f;
            for (int d = 0; d < D; d++)
            {
                qk_dot += param.q_ptr[qo_offset + d] * param.k_ptr[k_offset + d];
            }
            exp_sum += exp(qk_dot * param.scale - max_qk_dot);
        }

        // caclualte and write back
        // O (l_kv * d^2)
        for(int d = 0; d < D; d++)
        {
            float acc = 0;
            
            // reduce on l_kv
            for(int l_kv = 0; l_kv < L_kv; l_kv++)
            {
                int kv_offset = b_id * L_kv * D + l_kv * D;
                float qk_dot = 0;
                for (int di = 0; di < D; di++)
                {
                    qk_dot += param.q_ptr[qo_offset + di] * param.k_ptr[kv_offset + di];
                }
                float qk_dot_scaled = qk_dot * param.scale;
                float p = exp(qk_dot_scaled - max_qk_dot) / exp_sum; // output in [b, l_q, l_kv]
                acc += p * param.v_ptr[kv_offset + d];
            }
            param.o_ptr[qo_offset + d] = acc;
        }
    }
}


void launch_sdqa_attention_fwd_native(attention_param_t param)
{
    int b = param.batch;
    int lq = param.len_q;
    dim3 block(B, BQ);
    dim3 grid((b + B - 1) / B, (lq + BQ - 1) / BQ);
    sdqa_attention_fwd_native<<<grid, block>>>(param);

}