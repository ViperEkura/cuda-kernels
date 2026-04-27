#include "kernels/matmul.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int BM = 32;
static constexpr int BN = 32;

// MMA shape: m16n8k16
__device__ inline void mma_m16n8k16(uint32_t (&A)[4], uint32_t (&B)[2], float (&C)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]),
                   "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ inline void ldmatrix_a(uint32_t (&reg)[4], const void *smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

__device__ inline void ldmatrix_b(uint32_t (&reg)[2], const void *smem) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                 "{%0,%1}, [%2];"
                 : "=r"(reg[0]), "=r"(reg[1])
                 : "r"(addr));
}

__global__ void matmul_mma(matmul_param_t param)
{
    int N = param.N;
    int K = param.K;

    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;

    // warps arranged in 2x4 grid:
    // each warp compute 16x8 result matrix tile

    int nBlock = BN * blockIdx.x;
    int mBlock = BM * blockIdx.y;

    int WarpLd =  BN / 8;
    int nWarp = 8 * (warpID % WarpLd);
    int mWarp = 16 * (warpID / WarpLd);
    
    __shared__ half smemA[BM * 16]; // (BM, BK)
    __shared__ half smemB[BN * 16]; // (BK, BN)

    uint32_t regA[4];
    uint32_t regB[2];
    float regC[4] = {0.0f, 0.0f, 0.0f, 0.0f};


    for (int kOffset = 0; kOffset < K; kOffset += 16)
    {
        int nOffset = nBlock + nWarp;
        int mOffset = mBlock + mWarp;

        #pragma unroll
        for (int saddrA = laneID; saddrA < BM * 16; saddrA += 32)
        {
            int gaddrA = (mOffset + saddrA / 16) * K + kOffset + saddrA % 16;
            smemA[saddrA] = param.lhs[gaddrA];
        }

        #pragma unroll
        for (int saddrB = laneID; saddrB < BN * 16; saddrB += 32)
        {
            int gaddrB = (kOffset + saddrB / 8) * N + nOffset + saddrB % 8;
            smemB[saddrB] = param.rhs[gaddrB];
        }
        __syncthreads();

        ldmatrix_a(regA, smemA);
        ldmatrix_b(regB, smemB);
        mma_m16n8k16(regA, regB, regC);

        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) 
    {
        int r = laneID / 4 * 2 + i / 2;
        int c = laneID % 4 * 2 + i % 2;
        int global_r = mBlock + mWarp + r;
        int global_c = nBlock + nWarp + c;
        param.dst[global_r * N + global_c] = regC[i];
    }

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

    constexpr int wrap_num = (BM * BN) / (16 * 8);
    constexpr int wrap_size = 32;

    dim3 block(wrap_num * wrap_size);
    dim3 grid((param.N + BN - 1) / BN, (param.M + BM - 1) / BM);

    matmul_mma<<<grid, block>>>(param);
}