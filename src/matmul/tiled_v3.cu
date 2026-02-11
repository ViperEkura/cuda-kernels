#include "kernels/matmul.h"
#include "common.h"

static constexpr int BM = 64, BN = 64;
static constexpr int BK = 8;
static constexpr int TM = 8, TN = 8;
static constexpr int THREAD_NUM = (BM / TM) * (BN / TN);
#define SWIZZLE_BANK(x) ((x) ^ ((x) >> 5))

__global__ void matmul_tiled_v3(matmul_param_t param)
{
    const int M = param.M;
    const int N = param.N;
    const int K = param.K;
    const int tx = threadIdx.x;
    const int n_start = blockIdx.x * BN;
    const int m_start = blockIdx.y * BM;

    __shared__ float lhs[BK * BN];
    __shared__ float rhs[BK * BN];
    float dst[TM][TN] = {0};

    const int thread_row = tx / (BN / TN); 
    const int thread_col = tx % (BN / TN); 
    const int thread_m_start = thread_row * TM;
    const int thread_n_start = thread_col * TN;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++)
    {
        int k_start = bk * BK;

        for(int mk = tx; mk < BM * BK; mk += blockDim.x)
        {
            int load_lhs_m = mk / BK;
            int load_lhs_k = mk % BK;
            int load_smem_lhs_addr = mk;
            int load_gmem_lhs_addr = (m_start + load_lhs_m) * K + (k_start + load_lhs_k);
            if (m_start + load_lhs_m < M && k_start + load_lhs_k < K)
                lhs[SWIZZLE_BANK(load_smem_lhs_addr)] = __ldg(param.lhs + load_gmem_lhs_addr);
            else
                lhs[SWIZZLE_BANK(load_smem_lhs_addr)] = 0;
        }

        for(int kn = tx; kn < BK * BN; kn += blockDim.x)
        {
            int load_rhs_k = kn / BN; 
            int load_rhs_n = kn % BN; 
            int load_smem_rhs_addr = kn;
            int load_gmem_rhs_addr = (k_start + load_rhs_k) * N + (n_start + load_rhs_n);
            if(k_start + load_rhs_k < K && n_start + load_rhs_n < N) 
                rhs[SWIZZLE_BANK(load_smem_rhs_addr)] = __ldg(param.rhs + load_gmem_rhs_addr);
            else
                rhs[SWIZZLE_BANK(load_smem_rhs_addr)] = 0;
        }
        __syncthreads();

        for(int k = 0; k < BK; k++)
        {
            float lhs_reg[TM];
            for(int m = 0; m < TM; m++)
            {
                int lhs_m = thread_m_start + m;
                lhs_reg[m] = lhs[SWIZZLE_BANK(lhs_m * BK + k)];
            }
            float rhs_reg[TN];
            for(int n = 0; n < TN; n++)
            {
                int rhs_n = thread_n_start + n;
                rhs_reg[n] = rhs[SWIZZLE_BANK(k * BN + rhs_n)];
            }
            
            for(int m = 0; m < TM; m++)
            {
                for(int n = 0; n < TN; n++)
                {
                    dst[m][n] += lhs_reg[m] * rhs_reg[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
    {
        for(int n = 0; n < TN; n++)
        {
            int store_gmem_c_m = m_start + thread_m_start + m;
            int store_gmem_c_n = n_start + thread_n_start + n;

            if(store_gmem_c_m < M && store_gmem_c_n < N)
            {
                int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
                param.dst[store_gmem_c_addr] = dst[m][n];
            }
        }
    }
}


void launch_matmul_tiled_v3(matmul_param_t param)
{
    dim3 block(THREAD_NUM);  
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);
    matmul_tiled_v3<<<grid, block>>>(param);
}