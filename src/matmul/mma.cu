#include "kernels/matmul.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 16;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;

static constexpr int WM = 32;
static constexpr int WN = 32;

static constexpr int MMA_TILES_M = WM / MMA_M;
static constexpr int MMA_TILES_N = WN / MMA_N;

static constexpr int WARPS_M = BM / WM;
static constexpr int WARPS_N = BN / WN;
static constexpr int WARPS   = WARPS_M * WARPS_N;
static constexpr int THREADS = WARPS * 32;

static constexpr int B_SMEM_PER_WARP = MMA_TILES_N * BK * MMA_N;

#define PACK_HALF2(lo, hi) \
    (((uint32_t)*(uint16_t*)&(hi) << 16) | *(uint16_t*)&(lo))

__global__ void matmul_mma(matmul_param_t param)
{
    int M = param.M;
    int N = param.N;
    int K = param.K;

    float *lhs = param.lhs;
    float *rhs = param.rhs;
    float *dst = param.dst;

    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int wm_start = warp_m * WM;
    int wn_start = warp_n * WN;

    __shared__ half smemA[BM * BK];
    __shared__ half smemB[WARPS_N * B_SMEM_PER_WARP];

    float acc[MMA_TILES_M][MMA_TILES_N][4] = {};

    for (int k_block = 0; k_block < K; k_block += BK)
    {
        for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int g_row = block_m + row;
            int g_col = k_block + col;
            smemA[i] = (g_row < M && g_col < K)
                ? __float2half(lhs[g_row * K + g_col])
                : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < BN * BK; i += THREADS) {
            int k_row  = i / BN;
            int n_col  = i % BN;
            int g_k = k_block + k_row;
            int g_n = block_n + n_col;
            int tgt_warp_n  = n_col / WN;
            int col_in_warp = n_col % WN;
            int mma_tile    = col_in_warp / MMA_N;
            int col_in_mma  = col_in_warp % MMA_N;
            int smem_idx = tgt_warp_n * B_SMEM_PER_WARP
                         + mma_tile * (BK * MMA_N)
                         + k_row * MMA_N
                         + col_in_mma;
            smemB[smem_idx] = (g_k < K && g_n < N)
                ? __float2half(rhs[g_k * N + g_n])
                : __float2half(0.0f);
        }

        __syncthreads();

        int r0 = lane_id / 4;
        int k0 = (lane_id % 4) * 2;

#pragma unroll
        for (int mm = 0; mm < MMA_TILES_M; mm++) {
            int a_row0 = wm_start + mm * MMA_M + r0;
            int a_row1 = a_row0 + 8;

            uint32_t ra[4] = {
                PACK_HALF2(smemA[a_row0 * BK + k0],
                           smemA[a_row0 * BK + k0 + 1]),
                PACK_HALF2(smemA[a_row1 * BK + k0],
                           smemA[a_row1 * BK + k0 + 1]),
                PACK_HALF2(smemA[a_row0 * BK + 8 + k0],
                           smemA[a_row0 * BK + 8 + k0 + 1]),
                PACK_HALF2(smemA[a_row1 * BK + 8 + k0],
                           smemA[a_row1 * BK + 8 + k0 + 1]),
            };

#pragma unroll
            for (int mn = 0; mn < MMA_TILES_N; mn++) {
                int b_base = warp_n * B_SMEM_PER_WARP + mn * (BK * MMA_N);
                int kb0 = (lane_id % 4) * 2;
                int n0  = lane_id / 4;

                uint32_t rb[2] = {
                    PACK_HALF2(smemB[b_base +  kb0      * MMA_N + n0],
                               smemB[b_base + (kb0 + 1) * MMA_N + n0]),
                    PACK_HALF2(smemB[b_base + (kb0 + 8) * MMA_N + n0],
                               smemB[b_base + (kb0 + 9) * MMA_N + n0]),
                };

                float (&c)[4] = acc[mm][mn];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                    : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                      "r"(rb[0]), "r"(rb[1]),
                      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int mm = 0; mm < MMA_TILES_M; mm++) {
#pragma unroll
        for (int mn = 0; mn < MMA_TILES_N; mn++) {
            int r0 = lane_id / 4;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int row = block_m + wm_start + mm * MMA_M
                        + r0 + ((i >= 2) ? 8 : 0);
                int col = block_n + wn_start + mn * MMA_N
                        + (lane_id % 4) * 2 + (i & 1);
                if (row < M && col < N)
                    dst[row * N + col] = acc[mm][mn][i];
            }
        }
    }
}

void launch_matmul_mma(matmul_param_t param)
{
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

    if (major < 8) {
        fprintf(stderr,
            "MMA requires Compute Capability 8.0+ (SM80+). Current: SM%d%d\n",
            major, minor);
        return;
    }

    dim3 block(THREADS);
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);

    matmul_mma<<<grid, block>>>(param);
}
