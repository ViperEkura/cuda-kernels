#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

#include "kernels/matmul.h"

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;


__global__ void matmul_wmma(matmul_param_t param)
{
    int M = param.M;
    int N = param.N;
    int K = param.K;
    
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;

    __shared__ half A_shared[WMMA_M][WMMA_K];
    __shared__ half B_shared[WMMA_K][WMMA_N];

    const int num_tiles_k = (K + WMMA_K - 1) / WMMA_K;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        const int k_offset = tile_k * WMMA_K;
        
        for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += blockDim.x) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            if (tile_row + row < M && k_offset + col < K) {
                A_shared[row][col] = __float2half(param.lhs[(tile_row + row) * K + (k_offset + col)]);
            } else {
                A_shared[row][col] = __float2half(0.0f);
            }
        }
        
        for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            if (k_offset + row < K && tile_col + col < N) {
                B_shared[row][col] = __float2half(param.rhs[(k_offset + row) * N + (tile_col + col)]);
            } else {
                B_shared[row][col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
                
        wmma::load_matrix_sync(a_frag, &A_shared[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &B_shared[0][0], WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }
    
    if (tile_row < M && tile_col < N) {
        float* C_ptr = param.dst + tile_row * N + tile_col;
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