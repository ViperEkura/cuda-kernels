#include "kernels/matmul.h"
#include "common.h"

static constexpr int BM = 128, BN = 128;
static constexpr int BK = 8;
static constexpr int TM = 8, TN = 8;
static constexpr int THREAD_NUM = (BM / TM) * (BN / TN);
static constexpr int MEM_PER_THRED_LHS = (BM * BK) / THREAD_NUM;
static constexpr int MEM_PER_THRED_RHS = (BN * BK) / THREAD_NUM;
#define SWIZZLE_BANK(x) ((x) ^ ((x) >> 5))

__global__ void matmul_tiled_v3(matmul_param_t param)
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

    __shared__ float lhs[BK][BM];
    __shared__ float rhs[BK][BN];
    float dst[TM][TN] = {0};

    constexpr int VEC_SIZE = 4;
    static_assert(MEM_PER_THRED_LHS % VEC_SIZE == 0);
    static_assert(MEM_PER_THRED_RHS % VEC_SIZE == 0);

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        
        #pragma unroll
        for(int k = 0; k < MEM_PER_THRED_LHS; k += VEC_SIZE) {
            if(load_gmem_a_m < M && (load_gmem_a_k + k + 3) < K) {
                float4 vals = __ldg(reinterpret_cast<const float4*>(
                    param.lhs + load_gmem_a_addr + k
                ));
                
                lhs[load_smem_a_k + k + 0][SWIZZLE_BANK(load_smem_a_m)] = vals.x;
                lhs[load_smem_a_k + k + 1][SWIZZLE_BANK(load_smem_a_m)] = vals.y;
                lhs[load_smem_a_k + k + 2][SWIZZLE_BANK(load_smem_a_m)] = vals.z;
                lhs[load_smem_a_k + k + 3][SWIZZLE_BANK(load_smem_a_m)] = vals.w;
            } else {
                for(int i = 0; i < VEC_SIZE; i++) {
                    int k_idx = k + i;
                    if(load_gmem_a_m < M && (load_gmem_a_k + k_idx) < K) {
                        lhs[load_smem_a_k + k_idx][SWIZZLE_BANK(load_smem_a_m)] = 
                            param.lhs[load_gmem_a_addr + k_idx];
                    } else {
                        lhs[load_smem_a_k + k_idx][SWIZZLE_BANK(load_smem_a_m)] = 0.0f;
                    }
                }
            }
        }

        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        
        #pragma unroll
        for(int n = 0; n < MEM_PER_THRED_RHS; n += VEC_SIZE) {
            if(load_gmem_b_k < K && (load_gmem_b_n + n + 3) < N) {
                float4 vals = __ldg(reinterpret_cast<const float4*>(
                    param.rhs + load_gmem_b_addr + n
                ));
                
                rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n + 0)] = vals.x;
                rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n + 1)] = vals.y;
                rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n + 2)] = vals.z;
                rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n + 3)] = vals.w;
            } else {
                for(int i = 0; i < VEC_SIZE; i++) {
                    int n_idx = n + i;
                    if(load_gmem_b_k < K && (load_gmem_b_n + n_idx) < N) {
                        rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n_idx)] = 
                            param.rhs[load_gmem_b_addr + n_idx];
                    } else {
                        rhs[load_smem_b_k][SWIZZLE_BANK(load_smem_b_n + n_idx)] = 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();


        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float lhs_reg[TM];
            float rhs_reg[TN];

            #pragma unroll
            for(int m = 0; m < TM; m++) {
                int store_smem_a_m = ty * TM + m;
                lhs_reg[m] = lhs[k][SWIZZLE_BANK(store_smem_a_m)];
            }
            
            #pragma unroll
            for(int n = 0; n < TN; n++) {
                int store_smem_b_n = tx * TN + n;
                rhs_reg[n] = rhs[k][SWIZZLE_BANK(store_smem_b_n)];
            }
            
            #pragma unroll
            for(int m = 0; m < TM; m++) {
                #pragma unroll
                for(int n = 0; n < TN; n++) {
                    dst[m][n] += lhs_reg[m] * rhs_reg[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int store_gmem_c_m = by * BM + ty * TM + m;
        #pragma unroll
        for(int n = 0; n < TN; n++) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            if(store_gmem_c_m < M && store_gmem_c_n < N) {
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