#include "kernels/matmul.h"

static constexpr int BM = 128, BN = 128;
static constexpr int BK = 8;
static constexpr int TM = 8, TN = 8;

static constexpr int THREAD_NUM = (BM / TM) * (BN / TN);
static constexpr int MEM_PER_THRED_LHS = (BM * BK) / THREAD_NUM;
static constexpr int MEM_PER_THRED_RHS = (BN * BK) / THREAD_NUM;


__global__ void matmul_tiled_v2(matmul_param_t param)
{
    const int M = param.M;
    const int N = param.N;
    const int K = param.K;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int load_smem_a_m = tid / (BK / MEM_PER_THRED_LHS);
    const int load_smem_a_k = tid % (BK / MEM_PER_THRED_LHS) * MEM_PER_THRED_LHS;

    const int load_smem_b_k = tid / (BN / MEM_PER_THRED_RHS);
    const int load_smem_b_n = tid % (BN / MEM_PER_THRED_RHS) * MEM_PER_THRED_RHS;

    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;

    __shared__ float lhs[BM][BK];
    __shared__ float rhs[BK][BN];
    float dst[TM][TN];

    for(int m = 0; m < TM; m++){
        for(int n = 0; n < TN; n++){
            dst[m][n] = 0;
        }
    }
    __syncthreads();

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++){
        // step 1:  fetch
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    #pragma unroll
        for(int k = 0; k < MEM_PER_THRED_LHS; k++){
            if(load_gmem_a_m < M && (load_gmem_a_k + k) < K) {
                lhs[load_smem_a_m][load_smem_a_k + k] = param.lhs[load_gmem_a_addr + k];
            } else {
                lhs[load_smem_a_m][load_smem_a_k + k] = 0.0f;
            }
        }

    #pragma unroll
        for(int n = 0; n < MEM_PER_THRED_RHS; n++){
            if(load_gmem_b_k < K && (load_gmem_b_n + n) < N) {
                rhs[load_smem_b_k][load_smem_b_n + n] = param.rhs[load_gmem_b_addr + n];
            } else {
                rhs[load_smem_b_k][load_smem_b_n + n] = 0.0f;
            }
        }
        __syncthreads();

        // step 2: calculate
    #pragma unroll
        for (int k = 0; k < BK; k++){
    #pragma unroll
            for(int m = 0; m < TM; m++){
    #pragma unroll
                for(int n = 0; n < TN; n++){
                    int store_smem_a_m = ty * TM + m;
                    int store_smem_b_n = tx * TN + n;
                    dst[m][n] += lhs[store_smem_a_m][k] * rhs[k][store_smem_b_n];
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < TM; m++){
        int store_gmem_c_m = by * BM + ty * TM + m;
    #pragma unroll
        for(int n = 0; n < TN; n++){
            int store_gmem_c_n = bx * BN + tx * TN + n;
            if(store_gmem_c_m < M && store_gmem_c_n < N){
                int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
                param.dst[store_gmem_c_addr] = dst[m][n];
            }
        }
    }
}

void launch_matmul_tiled_v2(matmul_param_t param)
{
    dim3 block(BN / TN, BM / TM);  
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);
    matmul_tiled_v2<<<grid, block>>>(param);
}