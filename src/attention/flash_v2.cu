#include "kernels/attention.h"

static constexpr int Bl = 16;
static constexpr int Bd = 16;
static constexpr int Offset = 1;
// Bd >= Bl

__device__ float warp_reduce_max(float val, int range) {
    int group_size = range;
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int group_start = (lane_id / group_size) * group_size;
    unsigned group_mask = ((1u << group_size) - 1) << group_start;
    
    for (int offset = group_size / 2; offset > 0; offset >>= 1) 
        val = max(val, __shfl_xor_sync(group_mask, val, offset));

    return val;
}


__device__ float warp_reduce_sum(float val, int range) {
    int group_size = range;
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int group_start = (lane_id / group_size) * group_size;
    unsigned group_mask = ((1u << group_size) - 1) << group_start;
    
    for (int offset = group_size / 2; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(group_mask, val, offset);
    
    float sum = __shfl_sync(group_mask, val, group_start);
    return sum;
}


__global__ void sdqa_attention_fwd_flash_v2(attention_param_t param)
{
    const int D = param.dim;
    const int L_q = param.len_q;
    const int L_kv = param.len_kv;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int Tkv = (L_kv + Bl - 1) / Bl;
    const int Td = (D + Bd -1) / Bd;
    const float scale = param.scale;

    const int batch  = bz;
    const int tid = ty * blockDim.x + tx;
    const int load_gmem_l = by * blockDim.y + ty;
    const int load_gmem_d = bx * blockDim.x + tx;
    const int load_smem_l = ty;
    const int load_smem_d = tx;

    __shared__ float smem_q[Bd][Bl + Offset];
    __shared__ float smem_k[Bd][Bl + Offset];
    __shared__ float smem_v[Bd][Bl + Offset];
    __shared__ float smem_p[Bl][Bl + Offset];

    float row_max = -INFINITY;
    float row_sum = 0;
    double reg_o = 0;

    for (int kv_stride = 0; kv_stride < Tkv; kv_stride++)
    {
        int kv_start = kv_stride * Bl;
        float local_qk_sum = 0;
        float local_p = 0;

        // load V
        int load_smem_kv_l = load_smem_l;
        int load_gmem_kv_l = kv_start + load_smem_kv_l;
        int load_gmem_kv_d = load_gmem_d;
        int load_gmem_kv_addr = batch * L_kv * D + load_gmem_kv_l * D + load_gmem_kv_d;
        
        if (load_gmem_kv_l < L_kv && load_gmem_kv_d < D)
            smem_v[load_smem_d][load_smem_kv_l] = param.v_ptr[load_gmem_kv_addr];
        else
            smem_v[load_smem_d][load_smem_kv_l] = 0;
        

        for(int d_stride = 0; d_stride < Td; d_stride++)
        {
            int d_start = d_stride * Bd;
            int load_gmem_q_l = load_gmem_l;
            int load_gmem_q_d = d_start + load_smem_d;
            int load_gmem_q_addr = batch * L_q * D + load_gmem_q_l * D + load_gmem_q_d;
            // load Q, K
            if (load_gmem_q_l < L_q && load_gmem_q_d < D)
                smem_q[load_smem_d][load_smem_l] = param.q_ptr[load_gmem_q_addr];
            else 
                smem_q[load_smem_d][load_smem_l] = 0;

            int load_smem_kv_l = load_smem_l;
            int load_gmem_kv_l = kv_start + load_smem_kv_l;
            int load_gmem_kv_d = d_start + load_smem_d;
            int load_gmem_kv_addr = batch * L_kv * D + load_gmem_kv_l * D + load_gmem_kv_d;
            
            if (load_gmem_kv_l < L_kv && load_gmem_kv_d < D)
                smem_k[load_smem_d][load_smem_l] = param.k_ptr[load_gmem_kv_addr];
            else
                smem_k[load_smem_d][load_smem_l] = 0;

            __syncthreads();

            // S = Q @ K.T
            for(int d = 0; d < Bd; d++)
            {
                local_qk_sum += smem_q[d][tid / Bl] * smem_k[d][tid % Bl] * scale;
            }
            __syncthreads();
        }

        float block_max = -INFINITY;
        float block_sum = 0;

        // block_max = max(S, dim=-1)
        block_max = warp_reduce_max(local_qk_sum, Bl);
        // P = exp(S - block_max)
        local_p = exp(local_qk_sum - block_max);
        smem_p[tid / Bl][tid % Bl] = local_p;
        // block_sum = sum(P, dim=-1)
        block_sum = warp_reduce_sum(local_p, Bl);
        __syncthreads();

        float new_max = max(block_max, row_max);
        float old_scale = exp(row_max - new_max);
        float new_scale = exp(block_max - new_max);
        float pv = 0;

        // PV = P @ V
        for(int kv = 0; kv < Bl; kv++)
        {
            pv += smem_p[load_smem_l][kv] * smem_v[load_smem_d][kv];
        }
        // O = O * old_scale + PV * new_scale
        reg_o = reg_o * old_scale + pv * new_scale;
        row_sum = row_sum * old_scale + block_sum * new_scale;
        row_max = new_max;
        __syncthreads();
    }

    int load_gmem_o_l = load_gmem_l;
    int load_gmem_o_d = load_gmem_d;
    int load_gmem_o_addr = batch * L_q * D + load_gmem_o_l * D + load_gmem_o_d;

    if (load_gmem_o_l < L_q && load_gmem_o_d < D)
    {
        param.o_ptr[load_gmem_o_addr] = (float)(reg_o / (row_sum + param.eps));
    }
    
}

void launch_sdqa_attention_fwd_flash_v2(attention_param_t param)
{
    dim3 block(Bd, Bl);
    dim3 grid(
        (param.dim + Bd - 1) / Bd,
        (param.len_q + Bl - 1) / Bl, 
        param.batch
    );
    sdqa_attention_fwd_flash_v2<<<grid, block>>>(param);
}