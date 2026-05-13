#include "kernels/matmul.h"
#include "registry.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int BM = 128;
static constexpr int BN = 64;
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

static constexpr int BLK = 8;
static constexpr int BLK_SZ = 64;
static constexpr int BK_BLKS = BK / BLK;

static constexpr int B_TILE_SZ = BK_BLKS * BLK_SZ;
static constexpr int B_SMEM_PER_WARP = MMA_TILES_N * B_TILE_SZ;

__device__ __forceinline__ void mma_m16n8k16(
    uint32_t (&regA)[4], uint32_t (&regB)[2], float (&regC)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(regC[0]), "=f"(regC[1]), "=f"(regC[2]), "=f"(regC[3])
        : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
          "r"(regB[0]), "r"(regB[1]),
          "f"(regC[0]), "f"(regC[1]), "f"(regC[2]), "f"(regC[3]));
}

__device__ __forceinline__ void ldmatrix_a(
    uint32_t (&ra)[4], const half *smemA,
    int wm_start, int mm, int lane_id)
{
    int group     = lane_id >> 3;
    int row_local = lane_id & 7;

    int row_off = (group == 1 || group == 3) ? 8 : 0;
    int col_off = (group >= 2) ? 8 : 0;

    int row = wm_start + mm * MMA_M + row_off + row_local;
    int block_row = row / BLK;
    int inner    = row % BLK;
    int block_col = col_off / BLK;
    const half *addr = &smemA[(block_row * BK_BLKS + block_col) * BLK_SZ
                            + inner * BLK];

    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(saddr));
}

__device__ __forceinline__ void ldmatrix_b(
    uint32_t (&rb)[2], const half *smemB,
    int b_base, int lane_id)
{
    int group     = lane_id >> 3;
    int row_mod   = lane_id & 7;
    int block_off = group * BLK_SZ;

    const half *addr = &smemB[b_base + block_off + row_mod * BLK];

    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"(saddr));
}

__device__ __forceinline__ void store_regs_c(
    float *dst, int M, int N, int row, int col, float (&acc)[4])
{
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row + ((i >= 2) ? 8 : 0);
        int c = col + (i & 1);
        if (r < M && c < N)
            dst[r * N + c] = acc[i];
    }
}

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

            int bm = row / BLK, bk = col / BLK;
            int lr = row % BLK, lc = col % BLK;
            int idx = (bm * BK_BLKS + bk) * BLK_SZ + lr * BLK + lc;

            int g_row = block_m + row;
            int g_col = k_block + col;
            smemA[idx] = (g_row < M && g_col < K)
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

            int bk = k_row / BLK;
            int lk = k_row % BLK;
            int idx = tgt_warp_n * B_SMEM_PER_WARP
                    + mma_tile * B_TILE_SZ
                    + bk * BLK_SZ
                    + lk * BLK
                    + col_in_mma;

            smemB[idx] = (g_k < K && g_n < N)
                ? __float2half(rhs[g_k * N + g_n])
                : __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll
        for (int mm = 0; mm < MMA_TILES_M; mm++) {
            uint32_t ra[4];
            ldmatrix_a(ra, smemA, wm_start, mm, lane_id);

#pragma unroll
            for (int mn = 0; mn < MMA_TILES_N; mn++) {
                int b_base = warp_n * B_SMEM_PER_WARP + mn * B_TILE_SZ;
                uint32_t rb[2];
                ldmatrix_b(rb, smemB, b_base, lane_id);

                mma_m16n8k16(ra, rb, acc[mm][mn]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int mm = 0; mm < MMA_TILES_M; mm++) {
#pragma unroll
        for (int mn = 0; mn < MMA_TILES_N; mn++) {
            int row = block_m + wm_start + mm * MMA_M + (lane_id / 4);
            int col = block_n + wn_start + mn * MMA_N + (lane_id % 4) * 2;
            store_regs_c(dst, M, N, row, col, acc[mm][mn]);
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

REGISTER_KERNEL(mma, launch_matmul_mma)
