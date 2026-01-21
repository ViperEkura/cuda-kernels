#include "kernels/matmul.h"

static constexpr int TILE_SIZE = 32;

__global__ void matmul_tiled(matmul_param_t param)
{
    int M = param.M;
    int N = param.N;
    int K = param.K;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int load_gmem_m = by * TILE_SIZE + tid / TILE_SIZE;
    int load_gmem_n = bx * TILE_SIZE + tid % TILE_SIZE;

    int load_smem_a_m = tid / TILE_SIZE;
    int load_smem_a_k = tid % TILE_SIZE;
    int load_smem_b_k = tid / TILE_SIZE;
    int load_smem_b_n = tid % TILE_SIZE;

    int load_smem_a_addr = load_smem_a_m * TILE_SIZE + load_smem_a_k;
    int load_smem_b_addr = load_smem_b_k * TILE_SIZE + load_smem_b_n;
    int bk_stride_num = (K + TILE_SIZE - 1)/ TILE_SIZE;

    __shared__ float smem_lhs[TILE_SIZE * TILE_SIZE];
    __shared__ float smem_rhs[TILE_SIZE * TILE_SIZE];
    float sum = 0;

    for (int bk = 0; bk < bk_stride_num;  bk++)
    {
        int load_gmem_a_k = bk * TILE_SIZE + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_m * K + load_gmem_a_k;

        if (load_gmem_a_k < K && load_gmem_m < M)
            smem_lhs[load_smem_a_addr] = param.lhs[load_gmem_a_addr];
        else
            smem_lhs[load_smem_a_addr] = 0;

        int load_gmem_b_k = bk * TILE_SIZE + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_n;
        
        if (load_gmem_b_k < K && load_gmem_n < N)
            smem_rhs[load_smem_b_addr] = param.rhs[load_gmem_b_addr];
        else
            smem_rhs[load_smem_b_addr] = 0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++){
            sum += smem_lhs[load_smem_a_m * TILE_SIZE + k] * smem_rhs[k * TILE_SIZE + load_smem_b_n];
        }
        __syncthreads();
    }

    if (load_gmem_m < M || load_gmem_n < N)
    {
        int store_gmem_c_addr = load_gmem_m * N + load_gmem_n;
        param.dst[store_gmem_c_addr] = sum;
    }

}

void launch_matmul_tiled(matmul_param_t param)
{
    dim3 block(TILE_SIZE, TILE_SIZE);  
    dim3 grid((param.N + TILE_SIZE - 1) / TILE_SIZE, (param.M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<<<grid, block>>>(param);
}