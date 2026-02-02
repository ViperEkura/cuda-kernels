#include <cooperative_groups.h>
#include "kernels/attention.h"

static constexpr int Bq = 16;
static constexpr int Bkv = 32;
static constexpr int Bd = 16;
static constexpr int KV_MEM_PER_THRED = Bkv / Bq;
static constexpr int SP_MEM_PER_THRED = Bkv / Bd;


__global__ void sdqa_attention_fwd_flash_v1(attention_param_t param)
{
    const int B   = param.batch;
    const int Lq  = param.len_q;
    const int Lkv = param.len_kv;
    const int D   = param.dim;
    const int Tkv =  (param.len_kv + Bkv - 1) / Bkv;
    const int Tq  =  (param.len_q + Bq - 1) / Bq;
    const int Td  =  (param.dim + Bd - 1) / Bd;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const int batch = bz;
    const int tid = ty * blockDim.x + tx;

    const int load_smem_q_d = tid / Bq;
    const int load_smem_q_l = tid % Bq;
    const int load_smem_s_q = tid / (Bkv / SP_MEM_PER_THRED);
    const int load_smem_s_k = tid % (Bkv / SP_MEM_PER_THRED) * SP_MEM_PER_THRED;

    __shared__ float smem_q[Bd][Bq];
    __shared__ float smem_k[Bd][Bkv];
    __shared__ float smem_v[Bd][Bkv];
    __shared__ float smem_s[Bq][Bkv];

    float row_max = -INFINITY;
    float row_sum = 0;

    for (int kv_tile = 0; kv_tile < Tkv; kv_tile++)
    {
        int kv_start = kv_tile * Bkv;
        int load_smem_kv_d = tid / (Bkv / KV_MEM_PER_THRED);
        int load_smem_kv_l = tid % (Bkv / KV_MEM_PER_THRED) * KV_MEM_PER_THRED;

        for (int q_tile = 0; q_tile < Tq; q_tile++)
        {
            int q_start = q_tile * Bq;

            // set S = 0
            for (int i = 0; i < SP_MEM_PER_THRED; i++){
                for (int d = 0; d < D; d++) {
                    smem_s[load_smem_s_q][load_smem_s_k + i] = 0;
                }
            }
            __syncthreads();

            for(int d_tile = 0; d_tile < Td; d_tile++)
            {
                int d_start = d_tile * Bd;
                //laod K
                for (int i = 0; i < KV_MEM_PER_THRED; i++)
                {
                    int load_gmem_kv_addr = batch*Lkv*D + (load_smem_kv_l + i + kv_start)*D + d_start + load_smem_kv_d;
                    if (load_smem_kv_l + i + kv_start < Lq && d_start + load_smem_kv_d < D){
                        smem_k[load_smem_kv_d][load_smem_kv_l + i] = param.k_ptr[load_gmem_kv_addr];
                    }
                    else{
                        smem_k[load_smem_kv_d][load_smem_kv_l + i] = 0;
                    }
                }

                // load Q
                int load_gmem_q_addr = batch*Lq*D + (load_smem_q_l + q_start)*D + d_start + load_smem_q_d;
                if (load_smem_q_l + q_start < Lq && d_start + load_smem_q_d < D){
                    smem_q[load_smem_q_d][load_smem_q_l] = param.q_ptr[load_gmem_q_addr];
                }
                else{
                    smem_q[load_smem_q_d][load_smem_q_l] = 0;
                }

                // S = Q @ K.T
                float s_val = 0.0f;
                for (int i = 0; i < KV_MEM_PER_THRED; i++){
                    for (int d = 0; d < Bd; d++) {
                        s_val += smem_q[d][load_smem_s_q] * smem_k[d][load_smem_s_k + i];
                    }
                    smem_s[load_smem_s_q][load_smem_s_k + i] += s_val * param.scale;
                }
                __syncthreads();
            }

            float row_max_block = -INFINITY;
            float row_sum_bock = 0;

            // m = max(S, dim=-1)
            for(int i = 0; i < Bkv; i++)
            {
                row_max_block = fmaxf(row_max_block, smem_s[load_smem_s_q][i]);
            }
            __syncthreads();
            // P = exp(S - row_max)
            for (int i = 0; i < SP_MEM_PER_THRED; i++) {
                smem_s[load_smem_s_q][load_smem_s_k + i] = exp(smem_s[load_smem_s_q][load_smem_s_k + i] - row_max_block);
            }
            __syncthreads();

            // l = sum(S, dim=-1)
            for(int i = 0; i < Bkv; i++)
            {
                row_sum_bock += smem_s[load_smem_s_q][i];
            }
            __syncthreads();

            float new_max = max(row_max, row_max_block);
            float old_scale = exp(row_max - new_max);
            float new_scale = exp(row_max_block - new_max);

            row_max = new_max;
            row_sum = row_sum * old_scale + row_sum_bock * new_scale;

            for(int d_tile = 0; d_tile < Td; d_tile++)
            {
                int d_start = d_tile * Bd;
                for (int i = 0; i < KV_MEM_PER_THRED; i++)
                {
                    int load_gmem_kv_addr = batch*Lkv*D + (load_smem_kv_l + kv_start + i)*D + d_start + load_smem_kv_d;
                    if (load_gmem_kv_addr < B * Lkv * D){
                        smem_v[load_smem_kv_d][load_smem_kv_l + i] = param.v_ptr[load_gmem_kv_addr];
                    }
                    else{
                        smem_v[load_smem_kv_d][load_smem_kv_l + i] = 0;
                    }
                }
                float pv = 0;

                // PV = P @ V
                for(int i = 0; i < Bkv; i++)
                {
                    pv += smem_s[load_smem_s_q][i] * smem_v[load_smem_q_d][i];
                }

                int store_gmem_o_addr = batch*Lq*D + (load_smem_q_l + q_start)*D + d_start + load_smem_q_d;
                if(load_smem_q_l + q_start < Lq &&  d_start + load_smem_q_d < D)
                {
                    param.o_ptr[store_gmem_o_addr] = param.o_ptr[store_gmem_o_addr] / row_sum;
                }
            }
        }
    }

    // reduce
    for (int q_tile = 0; q_tile < Tq; q_tile++)
    {
        int q_start = q_tile * Bq;
        for(int d_tile = 0; d_tile < Td; d_tile++)
        {
            int d_start = d_tile * Bd;
            int store_gmem_o_addr = batch*Lq*D + (load_smem_q_l + q_start)*D + d_start + load_smem_q_d;
            if(load_smem_q_l + q_start < Lq &&  d_start + load_smem_q_d < D)
            {
                param.o_ptr[store_gmem_o_addr] = param.o_ptr[store_gmem_o_addr] / row_sum;
            }
        }
    }
}

void launch_sdqa_attention_fwd_flash_v1(attention_param_t param)
{
    dim3 block(Bd, Bq);
    dim3 grid(
        (param.dim + Bd - 1) / Bd,
        (param.len_q + Bq - 1) / Bq, 
        param.batch
    );
    sdqa_attention_fwd_flash_v1<<<grid, block>>>(param);
}