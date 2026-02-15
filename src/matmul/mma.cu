#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

#include "kernels/matmul.h"

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

#define FLOAT4_REF(x)(*reinterpret_cast<float4*>((x)))
#define HALF2_PTR(x)(reinterpret_cast<half2*>((x)))

__global__ void matmul_wmma(matmul_param_t param)
{
    int M = param.M;
    int N = param.N;
    int K = param.K;
    
    const int m_start = blockIdx.y * WMMA_M;
    const int n_start = blockIdx.x * WMMA_N;
    const int num_tiles_k = (K + WMMA_K - 1) / WMMA_K;

    __shared__ half A_shared[WMMA_M * WMMA_K];
    __shared__ half B_shared[WMMA_K * WMMA_N];
    
    float4 lhs_vec, rhs_vec;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        const int k_offset = tile_k * WMMA_K;
        
        for (int i = threadIdx.x; i < WMMA_M * WMMA_K / 4; i += blockDim.x) {
            int base_idx = i * 4;
            int row_a = base_idx / WMMA_K;
            int col_a = base_idx % WMMA_K;
            
            if (m_start + row_a < M && k_offset + col_a < K && col_a + 3 < WMMA_K) {
                float* src_ptr = param.lhs + (m_start + row_a) * K + (k_offset + col_a);
                lhs_vec = FLOAT4_REF(src_ptr);
                
                float2 f2_0 = make_float2(lhs_vec.x, lhs_vec.y);
                float2 f2_1 = make_float2(lhs_vec.z, lhs_vec.w);
                
                half2 h2_0 = __float22half2_rn(f2_0);
                half2 h2_1 = __float22half2_rn(f2_1);
                
                HALF2_PTR(A_shared)[base_idx/2] = h2_0;
                HALF2_PTR(A_shared)[base_idx/2 + 1] = h2_1;
            }
            
            int row_b = base_idx / WMMA_N;
            int col_b = base_idx % WMMA_N;
            if (k_offset + row_b < K && n_start + col_b < N && col_b + 3 < WMMA_N) {
                float* src_ptr = param.rhs + (k_offset + row_b) * N + (n_start + col_b);
                rhs_vec = FLOAT4_REF(src_ptr);
                
                float2 f2_0 = make_float2(rhs_vec.x, rhs_vec.y);
                float2 f2_1 = make_float2(rhs_vec.z, rhs_vec.w);
                
                half2 h2_0 = __float22half2_rn(f2_0);
                half2 h2_1 = __float22half2_rn(f2_1);
                
                HALF2_PTR(B_shared)[base_idx/2] = h2_0;
                HALF2_PTR(B_shared)[base_idx/2 + 1] = h2_1;
            }
        }

        __syncthreads();
                
        wmma::load_matrix_sync(a_frag, A_shared, WMMA_K);
        wmma::load_matrix_sync(b_frag, B_shared, WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }
    
    if (m_start < M && n_start < N) {
        float* C_ptr = param.dst + m_start * N + n_start;
        wmma::store_matrix_sync(C_ptr, acc_frag, N, wmma::mem_row_major);
    }
}

void launch_matmul_mma(matmul_param_t param)
{
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    
    if (major < 7) {
        fprintf(stderr, "WMMA requires Compute Capability 7.0 or higher (SM70+). Current: SM%d%d\n", major, minor);
        return;
    }
    
    dim3 block(32, 1);
    dim3 grid(
        (param.N + WMMA_N - 1) / WMMA_N,
        (param.M + WMMA_M - 1) / WMMA_M
    );
    
    matmul_wmma<<<grid, block>>>(param);
}