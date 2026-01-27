#include "kernels/matmul.h"
#include "common.h"

void (*launch_func)(matmul_param_t) = launch_matmul_tiled_dbuf;

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

    CHECK(cudaMalloc((void**)&param.lhs, sizeof(float) * M * K));
    CHECK(cudaMalloc((void**)&param.rhs, sizeof(float) * N * K));
    CHECK(cudaMalloc((void**)&param.dst, sizeof(float) * M * N))

    CHECK(cudaMemcpy(param.lhs, host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.rhs, host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice));

    launch_matmul_verify(param);
    CHECK(cudaDeviceSynchronize())
    CHECK(cudaMemcpy(host_C_verify, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    launch_func(param);
    CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK(cudaMemcpy(host_C, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    check_result(N, host_C, host_C_verify);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(param.lhs));
    CHECK(cudaFree(param.rhs));
    CHECK(cudaFree(param.dst));

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_C_verify);

    return 0;
}