#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("Error: %s, Line: %d\n", __FILE__, __LINE__);                    \
        printf("Error code: %d, Reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-1);                                                               \
    }                                                                           \
}

#define TILE_SIZE 4

struct param_t
{
    float *lp;
    float *rp;
    float *op;
    int M;
    int N;
    int K;
};

__global__ void matmul_kernel(param_t param)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x; // M
    int y = threadIdx.y + blockIdx.y * blockDim.y; // K

    __shared__ float seg_lp[TILE_SIZE][TILE_SIZE];
    __shared__ float seg_rp[TILE_SIZE][TILE_SIZE];
    __shared__ float seg_op[TILE_SIZE][TILE_SIZE];

    int lp_bias = x * param.N;
    int rp_bias = y * param.K;

    for(int n_bias = 0; n_bias < param.N; n_bias += TILE_SIZE)
    {
        int n = n_bias + ty;
        
        seg_lp[ty][tx] = param.lp[lp_bias]; // [M][N]
    }

}


int main()
{
    float *host_lp, *host_rp, *host_op;
    float *device_lp, *device_rp, *device_op;
    
    int M = 4096, N = 4096, K = 4096;

    param_t param;

    host_lp = (float*)malloc(sizeof(float) * M * N);
    host_rp = (float*)malloc(sizeof(float) * N * K);
    host_lp = (float*)malloc(sizeof(float) * M * K);

    for (int i = 0; i < M * N; i++) host_lp[i] = (float)rand() / RAND_MAX; 
    for (int i = 0; i < N * K; i++) host_rp[i] = (float)rand() / RAND_MAX; 
    for (int i = 0; i < M * K; i++) host_op[i] = 0;

    CHECK(cudaMalloc((void**)&device_lp, sizeof(float) * M * N));
    CHECK(cudaMalloc((void**)&device_lp, sizeof(float) * N * K));
    CHECK(cudaMalloc((void**)&device_lp, sizeof(float) * M * K));
    CHECK(cudaMemcpy(device_lp, host_lp, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_rp, host_rp, sizeof(float) * N * K, cudaMemcpyHostToDevice));

    
    param.lp = device_lp;
    param.rp = device_rp;
    param.op = device_op;
    param.M = M;
    param.N = N;
    param.K = K;


    int tile_size = TILE_SIZE;
    int bx = (M + tile_size - 1) / tile_size;
    int by = (K + tile_size - 1) / tile_size;

    dim3 thread(tile_size, tile_size);
    dim3 block(bx, by);
    
    matmul_kernel<<<block, thread>>>(param);
    CHECK(cudaMemcpy(host_op, device_op, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(device_lp));
    CHECK(cudaFree(device_rp));
    CHECK(cudaFree(device_op));

    free(host_lp);
    free(host_rp);
    free(host_op);

    return 0;

}