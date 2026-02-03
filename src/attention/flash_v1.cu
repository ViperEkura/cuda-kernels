#include <cooperative_groups.h>
#include "kernels/attention.h"

static constexpr int Bq = 16;
static constexpr int Bkv = 16;
static constexpr int Bd = 16;
static constexpr int KV_PER_Q = Bkv / Bq;

// 32 > Bd >= Bkv >= Bq

__global__ void sdqa_attention_fwd_flash_v1(attention_param_t param)
{
    const int D = param.dim;
    const int L_q = param.len_q;
    const int L_kv = param.len_kv;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int Tkv = (L_kv + Bkv - 1) / Bkv;
    const int Td = (D + Bd -1) / Bd;
    const float scale = param.scale;

    const int batch  = bz;
    const int load_gmem_q = by * blockDim.y + ty;
    const int load_gmem_d = bx * blockDim.x + tx;
    const int load_smem_q = ty;
    const int load_smem_d = tx;
    const int tid = ty * blockDim.x + tx;

    __shared__ float smem_q[Bq][Bd];
    __shared__ float smem_k[Bkv][Bd];
    __shared__ float smem_v[Bkv][Bd];
    __shared__ float smem_sp[Bq][Bkv];

    float row_max = -INFINITY;
    float row_sum = 0;
    float reg_o = 0;

    for (int kv_stride = 0; kv_stride < Tkv; kv_stride++)
    {
        int kv_start = kv_stride * Bkv;
        // S = 0;
        if(tid < Bq * Bkv)
        {
            int local_q = tid / Bkv;
            int local_k = tid % Bkv;

            for(int d = 0; d < Bd; d++)
            {
                smem_sp[local_q][local_k] = 0;
            }
        }
        __syncthreads();

        for(int d_stride = 0; d_stride < Td; d_stride++)
        {
            int d_start = d_stride * Bd;
            int load_gmem_q_l = load_gmem_q;
            int load_gmem_q_d = d_start + load_smem_d;
            int load_gmem_q_addr = batch * L_q * D + load_gmem_q_l * D + load_gmem_q_d;
            // load Q, K
            if (load_gmem_q_l < L_q && load_gmem_q_d < D)
                smem_q[load_smem_q][load_smem_d] = param.q_ptr[load_gmem_q_addr];
            else 
                smem_q[load_smem_q][load_smem_d] = 0;

            for (int i = 0; i < KV_PER_Q; i++)
            {
                int load_smem_kv_l = load_smem_q * KV_PER_Q + i;
                int load_gmem_kv_l = kv_start + load_smem_kv_l;
                int load_gmem_kv_d = d_start + load_smem_d;
                int load_gmem_kv_addr = batch * L_kv * D + load_gmem_kv_l * D + load_gmem_kv_d;
                
                if (load_gmem_kv_l < L_kv && load_gmem_kv_d < D)
                    smem_k[load_smem_q * KV_PER_Q + i][load_smem_d] = param.k_ptr[load_gmem_kv_addr];
                else
                    smem_k[load_smem_q * KV_PER_Q + i][load_smem_d] = 0;
            }
            
            __syncthreads();

            // S = Q @ K.T
            if(tid < Bq * Bkv)
            {
                int local_q = tid / Bkv;
                int local_k = tid % Bkv;
                for(int d = 0; d < Bd; d++)
                {
                    smem_sp[local_q][local_k] += smem_q[local_q][d] * smem_k[local_k][d] * scale;
                }
            }
            __syncthreads();
        }

        float block_max = -INFINITY;
        float block_sum = 0;
        // block_max = max(S, dim=-1)
        for (int  i = 0; i < Bkv; i++)
        {
            block_max = max(block_max, smem_sp[load_smem_q][i]);
        }
        __syncthreads();

        // P = exp(S - block_max)
        if(tid < Bq * Bkv)
        {
            int local_q = tid / Bkv;
            int local_k = tid % Bkv;
            smem_sp[local_q][local_k] = exp(smem_sp[local_q][local_k] - block_max);
        }
        __syncthreads();

        // block_sum = sum(P, dim=-1)
        for (int  i = 0; i < Bkv; i++)
        {
            block_sum += smem_sp[load_smem_q][i];
        }

        float new_max = max(block_max, row_max);
        float old_scale = exp(row_max - new_max);
        float new_scale = exp(block_max - new_max);
        float pv = 0;
        
        // load V
        for (int i = 0; i < KV_PER_Q; i++)
        {
            int load_smem_kv_l = load_smem_q * KV_PER_Q + i;
            int load_gmem_kv_l = kv_start + load_smem_kv_l;
            int load_gmem_kv_d = load_gmem_d;
            int load_gmem_kv_addr = batch * L_kv * D + load_gmem_kv_l * D + load_gmem_kv_d;
            
            if (load_gmem_kv_l < L_kv && load_gmem_kv_d < D)
                smem_v[load_smem_kv_l][load_smem_d] = param.v_ptr[load_gmem_kv_addr];
            else
                smem_v[load_smem_kv_l][load_smem_d] = 0;
        }
        __syncthreads();
        // PV = P @ V
        for(int kv = 0; kv < Bkv; kv++)
        {
            pv += smem_sp[load_smem_q][kv] * smem_v[kv][load_smem_d];
        }
        __syncthreads();
        
        // O = O * old_scale + PV * new_scale
        reg_o = reg_o * old_scale + pv * new_scale;
        row_sum = row_sum * old_scale + block_sum * new_scale;
        row_max = new_max;
    }

    int load_gmem_q_l = load_gmem_q;
    int load_gmem_q_d = load_gmem_d;
    int load_gmem_q_addr = batch * L_q * D + load_gmem_q_l * D + load_gmem_q_d;

    if (load_gmem_q_l < L_q && load_gmem_q_d < D)
    {
        param.o_ptr[load_gmem_q_addr] = reg_o / (row_sum + param.eps);
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