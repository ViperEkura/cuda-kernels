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

struct param_t
{
    float *A;
    float *B;
    float  alpha;
    int    p_size;
};

__global__ void operator_kernel(param_t param)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = bx * blockDim.x + tx;
    if (idx >= param.p_size) return;

    param.A[idx] = param.A[idx] * param.alpha + param.B[idx];
}


int main()
{
    const int N = (int)1e9;
    const float alpha = 1;
    float *host_A, *host_B;
    
    param_t param;
    param.alpha = alpha;
    param.p_size = N;

    host_A = (float*)malloc(sizeof(float) * N);
    host_B = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; ++i)
        host_A[i] = host_B[i] = 1;

    CHECK(cudaMalloc((void**)&param.A, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&param.B, sizeof(float) * N));
    CHECK(cudaMemcpy(param.A, host_A, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.B, host_B, sizeof(float) * N, cudaMemcpyHostToDevice));

    int thread = 32;
    int block = (N + thread - 1) / thread;
    

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    operator_kernel<<<block, thread>>>(param);
    CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CHECK(cudaDeviceSynchronize()); // 同步
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // 销毁 CUDA 事件
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(host_A, param.A, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(host_B, param.B, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(param.A));
    CHECK(cudaFree(param.B));

    free(host_A);
    free(host_B);

    return 0;
}