#include "kernels/matmul.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int BM = 32;
static constexpr int BN = 32;
#define UINT32_PTR(x)(reinterpret_cast<uint32_t*>((x)))

__device__ inline void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float C[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]),
                   "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ inline void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}


__global__ void matmul_mma(matmul_param_t param)
{
    int N = param.N;
    int K = param.K;

    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;

    // warps arranged in 2x4 grid:
    // (warp_0 | warp_1 | warp_2 | warp_3)
    // (warp_4 | warp_5 | warp_6 | warp_7)

    // each warp compute 16x8 result matrix tile

    int nBlock = BN * blockIdx.x;
    int mBlock = BM * blockIdx.y;

    int WarpLd =  BN / 8;
    int nWarp = 8 * (warpID % WarpLd);
    int mWarp = 16 * (warpID / WarpLd);
    
    half A_frag[8];
    half B_frag[4];
    float C_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_start = 0; k_start < K; k_start += 16)
    {
        A_frag[0] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4)) * K + k_start + (laneID % 4) * 2]);
        A_frag[1] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4)) * K + k_start + (laneID % 4) * 2 + 1]);
        A_frag[2] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4)) * K + k_start + (laneID % 4) * 2 + 8]);
        A_frag[3] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4)) * K + k_start + (laneID % 4) * 2 + 9]);

        A_frag[4] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4) + 8) * K + k_start + (laneID % 4) * 2]);
        A_frag[5] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4) + 8) * K + k_start + (laneID % 4) * 2 + 1]);
        A_frag[6] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4) + 8) * K + k_start + (laneID % 4) * 2 + 8]);
        A_frag[7] = __float2half(param.lhs[(mBlock + mWarp + (laneID / 4) + 8) * K + k_start + (laneID % 4) * 2 + 9]);

        B_frag[0] = __float2half(param.rhs[(k_start + (laneID % 4) * 2) * N + nBlock + nWarp + (laneID / 4)]);
        B_frag[1] = __float2half(param.rhs[(k_start + (laneID % 4) * 2 + 1) * N + nBlock + nWarp + (laneID / 4)]);
        B_frag[2] = __float2half(param.rhs[(k_start + (laneID % 4) * 2 + 8) * N + nBlock + nWarp + (laneID / 4)]);
        B_frag[3] = __float2half(param.rhs[(k_start + (laneID % 4) * 2 + 9) * N + nBlock + nWarp + (laneID / 4)]);

        mma_m16n8k16(UINT32_PTR(A_frag), UINT32_PTR(B_frag), C_frag);
    }

    param.dst[(mBlock + mWarp + (laneID / 4)) * N + nBlock + nWarp + (laneID % 4) * 2] = C_frag[0];
    param.dst[(mBlock + mWarp + (laneID / 4)) * N + nBlock + nWarp + (laneID % 4) * 2 + 1] = C_frag[1];
    param.dst[(mBlock + mWarp + (laneID / 4) + 8) * N + nBlock + nWarp + (laneID % 4) * 2] = C_frag[2];
    param.dst[(mBlock + mWarp + (laneID / 4) + 8) * N + nBlock + nWarp + (laneID % 4) * 2 + 1] = C_frag[3];

}


void launch_matmul_mma(matmul_param_t param)
{
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    
    if (major < 7) {
        fprintf(stderr, "MMA requires Compute Capability 7.0 or higher (SM70+). Current: SM%d%d\n", major, minor);
        return;
    }

    constexpr int WARP_NUM = (BM * BN) / (16 * 8);
    constexpr int WARP_SIZE = 32;

    dim3 block(WARP_NUM * WARP_SIZE);
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);

    matmul_mma<<<grid, block>>>(param);
}