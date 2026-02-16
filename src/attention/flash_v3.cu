#include "kernels/attention.h"

static constexpr int Bl = 64;
static constexpr int Bd = 32;

#define FLOAT4_PTR(x)(reinterpret_cast<float4*>((x)))
#define FLOAT4_REF(x)(*reinterpret_cast<float4*>((x)))
#define SWIZZLE_BANK(x) ((((x >> 5) ^ (x >> 2)) << 2) + (x & 3))

__global__ void sdqa_attention_fwd_flash_v3(attention_param_t param)
{
    const int D   = param.dim;
    const int Lq  = param.len_q;
    const int Lkv = param.len_kv;

    const int Tkv = (param.len_kv + Bl - 1) / Bl; 
    const int Td  = (param.dim + Bd - 1) / Bd;
    const int tx  = threadIdx.x;

    const int batch   =  blockIdx.y;
    const int q_start =  blockIdx.x * blockDim.x;

    __shared__ float smem_q[Bl * Bd];
    __shared__ float smem_k[Bl * Bd];
    __shared__ float smem_v[Bl * Bd];

    float reg_o[Bd];
    float reg_s[Bl];

    for (int o_d_stride = 0; o_d_stride < Td; o_d_stride++)
    {
        int o_d_start = o_d_stride * Bl;
        float row_max = -INFINITY;
        float row_sum = 0;

        // O = 0
        for(int d = 0; d < Bd; d++)
        {
            reg_o[d] = 0;
        }

        for (int kv_stride = 0; kv_stride < Tkv; kv_stride++)
        {
            int kv_start = kv_stride * Bl;

            // load V
            for(int d = 0; d + 3 < Bd; d += 4)
            {
                int load_smem_v = tx * Bd + d;
                int load_gmem_v = batch * Lkv * D + (kv_start + tx) * D + (o_d_start + d);
                if (kv_start + tx < Lkv && o_d_start + d < D)
                    FLOAT4_REF(smem_v + SWIZZLE_BANK(load_smem_v)) = __ldg(FLOAT4_PTR(param.v_ptr + load_gmem_v));
                else
                    for(int td = 0; td < 4; td++)
                        smem_v[SWIZZLE_BANK(load_smem_v + td)] = 0;
            }
            // S = 0
            for(int kv = 0; kv < Bl; kv++)
            {
                reg_s[kv] = 0;
            }

            for(int qk_d_stride = 0; qk_d_stride < Td; qk_d_stride++)
            {
                int qk_d_start = qk_d_stride * Bd;
                // laod Q, K
                for(int d = 0; d + 3 < Bd; d += 4)
                {
                    int load_smem_q = tx * Bd + d;
                    int load_gmem_q = batch * Lq * D + (q_start + tx) * D + (qk_d_start + d);

                    if (q_start + tx < Lq && qk_d_start + d < D)
                        FLOAT4_REF(smem_q + SWIZZLE_BANK(load_smem_q)) = __ldg(FLOAT4_PTR(param.q_ptr + load_gmem_q));
                    else
                        for(int td = 0; td < 4; td++)
                            smem_q[SWIZZLE_BANK(load_smem_q + td)] = 0;

                    int load_smem_k = tx * Bd + d;
                    int load_gmem_k = batch * Lkv * D + (kv_start + tx) * D + (qk_d_start + d);

                    if (kv_start + tx < Lkv && qk_d_start + d < D)
                        FLOAT4_REF(smem_k + SWIZZLE_BANK(load_smem_k)) = __ldg(FLOAT4_PTR(param.k_ptr + load_gmem_k));
                    else
                        for(int td = 0; td < 4; td++)
                            smem_k[SWIZZLE_BANK(load_smem_k + td)] = 0;
                }

                __syncthreads();
                // S = Q @ K.T
                for(int d = 0; d < Bd; d++)
                {
                    int load_smem_q = tx * Bd + d;
                    float q_val = smem_q[SWIZZLE_BANK(load_smem_q)];
                    
                    for(int kv = 0; kv < Bl; kv++)
                    {
                        int load_smem_k = kv * Bd + d;
                        reg_s[kv] += q_val * smem_k[SWIZZLE_BANK(load_smem_k)];
                    }
                }
                __syncthreads();
            }
            for (int kv = 0; kv < Bl; kv++)
            {
                reg_s[kv] *= param.scale;
            }

            float block_max = -INFINITY;
            float block_sum = 0;

            // block_max = max(S, dim=-1)
            for (int kv = 0; kv < Bl; kv++)
            {
                block_max = max(block_max, reg_s[kv]);
            }
            // P = exp(S - block_max)
            // block_sum = sum(P, dim=-1)
            for (int kv = 0; kv < Bl; kv+=4)
            {
                reg_s[kv] = __expf(reg_s[kv] - block_max);
                reg_s[kv + 1] = __expf(reg_s[kv + 1] - block_max);
                reg_s[kv + 2] = __expf(reg_s[kv + 2] - block_max);
                reg_s[kv + 3] = __expf(reg_s[kv + 3] - block_max);

                block_sum += reg_s[kv];
                block_sum += reg_s[kv + 1];
                block_sum += reg_s[kv + 2];
                block_sum += reg_s[kv + 3];
            }

            float new_max = max(block_max, row_max);
            float old_scale = __expf(row_max - new_max);
            float new_scale = __expf(block_max - new_max);

            // O_acc = old_scale * O_acc + new_scale * P @ V
            for(int d = 0; d < Bd; d++)
            {
                float pv = 0;
                for(int kv = 0; kv < Bl; kv++)
                {
                    int load_smem_v = kv * Bd + d;
                    pv += reg_s[kv] * smem_v[SWIZZLE_BANK(load_smem_v)];
                }
                reg_o[d] = reg_o[d] * old_scale + pv * new_scale;
            }

            row_sum = row_sum * old_scale + block_sum * new_scale;
            row_max = new_max;
        }
        __syncthreads();

        // O = O_acc / row_sum
        for(int d = 0; d < Bd; d++)
        {
            int store_gmem_o = batch * Lq * D + (q_start + tx) * D + (o_d_start + d);
            if (q_start + tx < Lq && o_d_start + d < D)
            {
                param.o_ptr[store_gmem_o] = reg_o[d] / (row_sum + param.eps);
            }
        }
        __syncthreads();
    }
}


void launch_sdqa_attention_fwd_flash_v3(attention_param_t param)
{
    dim3 block(Bl);
    dim3 grid((param.len_q + Bl - 1) / Bl, param.batch);
    sdqa_attention_fwd_flash_v3<<<grid, block>>>(param);
}