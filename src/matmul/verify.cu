#include "kernels/matmul.h"

static constexpr int TILE_SIZE = 16;

__global__ void matmul_verify(matmul_param_t param)
{
    float* A = param.lhs;
    float* B = param.rhs;
    float* C = param.dst;

    int M = param.M;
    int N = param.N;
    int K = param.K;

    int tx = threadIdx.x,  ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row >= M || col >= N) return;
    
    float sum = 0;
    for (int i = 0; i < K; i++)
    {
        sum += A[row * K + i] * B[i *  N + col];
    }

    C[row * N + col] = sum;
}

void launch_matmul_verify(matmul_param_t param)
{
    dim3 block(TILE_SIZE, TILE_SIZE);  
    dim3 grid((param.N + TILE_SIZE - 1) / TILE_SIZE, (param.M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_verify<<<grid, block>>>(param);
}