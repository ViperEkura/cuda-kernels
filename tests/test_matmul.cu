#include "kernels/matmul.h"
#include "common.h"


int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    matmul_param_t param;
    param.M = M;
    param.N = N;
    param.K = K;


    float* host_A = (float*)malloc(sizeof(float) * M * K);
    float* host_B = (float*)malloc(sizeof(float) * N * K);
    float* host_C = (float*)malloc(sizeof(float) * M * N);

    CHECK(cudaMalloc((void**)&param.src_A, sizeof(float) * M * K));
    CHECK(cudaMalloc((void**)&param.src_B, sizeof(float) * N * K));
    CHECK(cudaMalloc((void**)&param.dst, sizeof(float) * M * N))

    CHECK(cudaMemcpy(param.src_A, host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.src_B, host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.dst, host_C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    launch_matmul_verify(param);
    CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CHECK(cudaDeviceSynchronize()); // 同步
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // 销毁 CUDA 事件
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(host_A, param.src_A, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(host_B, param.src_B, sizeof(float) * N * K, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(host_C, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost))
    CHECK(cudaFree(param.src_A));
    CHECK(cudaFree(param.src_B));
    CHECK(cudaFree(param.dst));

    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}