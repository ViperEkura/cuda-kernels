#include "kernels/matmul.h"
#include "common.h"

void (*launch_func)(matmul_param_t) = launch_matmul_tiled_v2;

float calcu_gflops(float m, float n, float k, float ms)
{
    float total_flops = 2 * m * n * k;
    //total_flops * 1000 / ms FLOPS;
    return total_flops / (ms * 1e6);
}

int main(int argc, char** argv)
{
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    
    matmul_param_t param;
    param.M = M;
    param.N = N;
    param.K = K;

    float* host_A = (float*)malloc(sizeof(float) * M * K);
    float* host_B = (float*)malloc(sizeof(float) * N * K);
    float* host_C = (float*)malloc(sizeof(float) * M * N);
    float* host_C_verify = (float*)malloc(sizeof(float) * M * N);

    unsigned seed = 42;
    srand(seed);
    for (int i = 0; i < M * K; i++) host_A[i] = (rand() % 255) / 256.0f;
    for (int i = 0; i < N * K; i++) host_B[i] = (rand() % 255) / 256.0f;

    CUDA_CHECK(cudaMalloc((void**)&param.lhs, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&param.rhs, sizeof(float) * N * K));
    CUDA_CHECK(cudaMalloc((void**)&param.dst, sizeof(float) * M * N))

    CUDA_CHECK(cudaMemcpy(param.lhs, host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(param.rhs, host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice));

    launch_matmul_cublas(param);
    CUDA_CHECK(cudaDeviceSynchronize())
    CUDA_CHECK(cudaMemcpy(host_C_verify, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch_func(param);
    CUDA_CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(host_C, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(M, N, K, milliseconds));
    check_result(N, host_C, host_C_verify);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(param.lhs));
    CUDA_CHECK(cudaFree(param.rhs));
    CUDA_CHECK(cudaFree(param.dst));

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_C_verify);

    return 0;
}