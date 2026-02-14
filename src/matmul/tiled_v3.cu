#include "kernels/matmul.h"
#include "common.h"

static constexpr int BM = 128, BN = 128;
static constexpr int BK = 8;
static constexpr int TM = 8, TN = 8;
static constexpr int THREAD_NUM = (BM / TM) * (BN / TN);
static constexpr int MEM_PER_THRED_LHS = (BM * BK) / THREAD_NUM; //BK * TM * TN / BN
static constexpr int MEM_PER_THRED_RHS = (BN * BK) / THREAD_NUM; //BK * TM * TN / BM

#define FLOAT4_PTR(x)(reinterpret_cast<float4*>((x)))
#define FLOAT4_REF(x)(*reinterpret_cast<float4*>((x)))
#define SWIZZLE_BANK(x) ((x) ^ ((x) >> 5))

__global__ void matmul_tiled_v3(matmul_param_t param)
{
    // check param
    static_assert(MEM_PER_THRED_LHS % 4 ==0 && MEM_PER_THRED_LHS / 4 > 0);
    static_assert(MEM_PER_THRED_RHS % 4 ==0 && MEM_PER_THRED_RHS / 4 > 0);

    const int M = param.M;
    const int N = param.N;
    const int K = param.K;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int m_start = by * BM;
    const int n_start = bx * BN;
    int load_smem_lhs_m = (MEM_PER_THRED_LHS * tid) / BK;
    int load_smem_lhs_k = (MEM_PER_THRED_LHS * tid) % BK;
    int load_smem_rhs_k = (MEM_PER_THRED_RHS * tid) / BN;
    int load_smem_rhs_n = (MEM_PER_THRED_RHS * tid) % BN;

    __shared__ float lhs[BK][BM];
    __shared__ float rhs[BK][BN];
    float dst[TM][TN] = {0};

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int k_start = bk * BK;
        int load_gmem_lhs_offset = m_start * K + k_start;
        int load_gmem_rhs_offset = k_start * N + n_start;
        int load_gmem_lhs_addr = load_gmem_lhs_offset + load_smem_lhs_m * K + load_smem_lhs_k;
        int load_gmem_rhs_addr = load_gmem_rhs_offset + load_smem_rhs_k * N + load_smem_rhs_n;

        float reg_lhs[MEM_PER_THRED_LHS];
        float reg_rhs[MEM_PER_THRED_RHS];

        // load
        for (int i = 0; i < MEM_PER_THRED_LHS / 4; i++)
        {
            FLOAT4_REF(&reg_lhs[i * 4]) = __ldg(FLOAT4_PTR(param.lhs + load_gmem_lhs_addr + i * 4));
        } 
        for (int i = 0; i < MEM_PER_THRED_RHS / 4; i++)
        {
            FLOAT4_REF(&reg_rhs[i * 4]) = __ldg(FLOAT4_PTR(param.rhs + load_gmem_rhs_addr + i * 4));
        }

        for (int g = 0; g < MEM_PER_THRED_LHS; g++)
        {
            int store_smem_m = (MEM_PER_THRED_LHS * tid + g) / BK;
            int store_smem_k = (MEM_PER_THRED_LHS * tid + g) % BK;
            int load_gmem_lhs_m = m_start + store_smem_m;
            int load_gmem_lhs_k = k_start + store_smem_k;
            reg_lhs[g] = (load_gmem_lhs_m < M && load_gmem_lhs_k < K) ? reg_lhs[g] : 0;
            lhs[store_smem_k][SWIZZLE_BANK(store_smem_m)] = reg_lhs[g];
        }

        for (int g = 0; g < MEM_PER_THRED_RHS; g++)
        {
            int store_smem_k = (MEM_PER_THRED_RHS * tid + g) / BN;
            int store_smem_n = (MEM_PER_THRED_RHS * tid + g) % BN;
            int load_gmem_rhs_k = k_start + load_smem_rhs_k;
            int load_gmem_rhs_n = n_start + load_smem_rhs_n + g;
            reg_rhs[g] = (load_gmem_rhs_k < K && load_gmem_rhs_n < N) ? reg_rhs[g] : 0;
            rhs[store_smem_k][SWIZZLE_BANK(store_smem_n)] = reg_rhs[g];
        }
        __syncthreads();

        // compute
        for (int k = 0; k < BK; k++) 
        {
            float lhs_reg[TM];
            float rhs_reg[TN];
            
            for (int m = 0; m < TM; m++) 
            {
                int comp_m = ty * TM + m;
                lhs_reg[m] = lhs[k][SWIZZLE_BANK(comp_m)];
            }
            
            for (int n = 0; n < TN; n++) 
            {
                int comp_n = tx * TN + n;
                rhs_reg[n] = rhs[k][SWIZZLE_BANK(comp_n)];
            }
            
            for (int m = 0; m < TM; m++) 
            {
                for (int n = 0; n < TN; n++) 
                {
                    dst[m][n] += lhs_reg[m] * rhs_reg[n];
                }
            }
        }
        __syncthreads();
    }

    // sotre
    for (int m = 0; m < TM; m++) 
    {
        for(int n = 0; n < TN; n++) 
        {
            int store_gmem_c_m = by * BM + ty * TM + m;
            int store_gmem_c_n = bx * BN + tx * TN + n;
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
    dim3 block(BN / TN, BM / TM);  
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);
    matmul_tiled_v3<<<grid, block>>>(param);
}