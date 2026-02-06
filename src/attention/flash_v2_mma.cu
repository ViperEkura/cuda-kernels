#include "kernels/attention.h"
#include <mma.h>


using namespace nvcuda;
static constexpr int Bl = 16;
static constexpr int Bd = 16;

__global__ void sdqa_attention_fwd_flash_v2_mma(attention_param_t param)
{
    const int D   = param.dim;
    const int Lq  = param.len_q;
    const int Lkv = param.len_kv;

    const int Tkv = (param.len_kv + Bl - 1) / Bl; 
    const int Td  = (param.dim + Bd - 1) / Bd;
    const int tx  = threadIdx.x;

    const int batch  =  blockIdx.y;
    const int seq_id =  blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ half smem_q[Bd * Bl];
    __shared__ half smem_k[Bd * Bl];
    __shared__ float smem_v[Bd * Bl];
    __shared__ float smem_s[Bl * Bl];
    __shared__ float smem_o[Bd * Bl];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    for (int o_d_stride = 0; o_d_stride < Td; o_d_stride++)
    {
        int o_d_start = o_d_stride * Bl;
        float row_max = -INFINITY;
        float row_sum = 0;

        // O = 0
        for(int kv = 0; kv < Bl; kv++)
        {
            smem_o[kv * Bl + tx] = 0;
        }

        for (int kv_stride = 0; kv_stride < Tkv; kv_stride++)
        {
            int kv_start = kv_stride * Bl;

            // load V
            for(int d = 0; d < Bd; d++)
            {
                int load_smem_v = d * Bl + tx;
                int load_gmem_v = batch * Lkv * D + (kv_start + tx) * D + (o_d_start + d);
                if (kv_start + tx < Lkv && o_d_start + d < D)
                    smem_v[load_smem_v] = param.v_ptr[load_gmem_v];
                else
                    smem_v[load_smem_v] = 0;
            }
            // S = 0
            for(int kv = 0; kv < Bl; kv++)
            {
                smem_s[kv * Bl + tx] = 0;
            }

            wmma::fill_fragment(acc_frag, 0.0f);
            __syncthreads();

            for(int qk_d_stride = 0; qk_d_stride < Td; qk_d_stride++)
            {
                int qk_d_start = qk_d_stride * Bd;
                // laod Q, K
                for(int d = 0; d < Bd; d++)
                {
                    int load_smem_q = d * Bl + tx;
                    int load_gmem_q = batch * Lq * D + seq_id * D + (qk_d_start + d);

                    if (seq_id < Lq && qk_d_start + d < D)
                        smem_q[load_smem_q] = __float2half(param.q_ptr[load_gmem_q]);
                    else
                        smem_q[load_smem_q] = __float2half(0);

                    int load_smem_k = d * Bl + tx;
                    int load_gmem_k = batch * Lkv * D + (kv_start + tx) * D + (qk_d_start + d);

                    if (kv_start + tx < Lkv && qk_d_start + d < D)
                        smem_k[load_smem_k] = __float2half(param.k_ptr[load_gmem_k]);
                    else
                        smem_k[load_smem_k] = __float2half(0);
                }

                __syncthreads();

                // S = Q @ K.T
                wmma::load_matrix_sync(q_frag, smem_q, Bl);
                wmma::load_matrix_sync(k_frag, smem_k, Bl);
                wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

                __syncthreads();
            }

            wmma::store_matrix_sync(smem_s, acc_frag, Bl, wmma::mem_col_major);
             __syncthreads();

            float block_max = -INFINITY;
            float block_sum = 0;

            // block_max = max(S, dim=-1)
            for (int kv = 0; kv < Bl; kv++)
            {
                block_max = max(block_max, smem_s[kv * Bl + tx] * param.scale);
            }
            // P = exp(S - block_max)
            // block_sum = sum(P, dim=-1)
            for (int kv = 0; kv < Bl; kv++)
            {
                smem_s[kv * Bl + tx] = exp(smem_s[kv * Bl + tx] * param.scale - block_max);
                block_sum += smem_s[kv * Bl + tx];
            }
            __syncthreads();

            float new_max = max(block_max, row_max);
            float old_scale = exp(row_max - new_max);
            float new_scale = exp(block_max - new_max);

            // PV = P @ V
            float reg_pv[Bd];
            for(int d = 0; d < Bd; d++)
            {
                reg_pv[d] = 0;
                for(int kv = 0; kv < Bl; kv++)
                {
                    int load_smem_p = kv * Bl + tx;
                    int load_smem_v = d * Bl + kv;
                    reg_pv[d] += smem_s[load_smem_p] * smem_v[load_smem_v];
                }
            }
            for(int d = 0; d < Bd; d++)
            {
                int load_smem_o = d * Bl + tx;
                smem_o[load_smem_o] = smem_o[load_smem_o] * old_scale + reg_pv[d] * new_scale;

            }

            row_sum = row_sum * old_scale + block_sum * new_scale;
            row_max = new_max;
        }
        __syncthreads();

        for(int d = 0; d < Bd; d++)
        {
            int load_smem_o = d * Bl + tx;
            int store_gmem_o = batch * Lq * D + seq_id * D + (o_d_start + d);
            if (seq_id < Lq && o_d_start + d < D)
            {
                param.o_ptr[store_gmem_o] = smem_o[load_smem_o] / row_sum;
            }
        }
        __syncthreads();
    }
}


void launch_sdqa_attention_fwd_flash_v2_mma(attention_param_t param)
{
    dim3 block(Bl);
    dim3 grid((param.len_q + Bl - 1) / Bl, param.batch);
    sdqa_attention_fwd_flash_v2_mma<<<grid, block>>>(param);
}