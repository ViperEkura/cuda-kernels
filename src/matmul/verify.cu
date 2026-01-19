#include "kernels/matmul.h"

__global__ void matmul_verify(matmul_param_t param)
{
    float* A = param.src_A;
    float* B = param.src_B;
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
    dim3 block(16, 16);  
    dim3 grid((param.N + 15) / 16, (param.M + 15) / 16);
    matmul_verify<<<grid, block>>>(param);
}