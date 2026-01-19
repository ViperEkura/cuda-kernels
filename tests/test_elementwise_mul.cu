#include "elementwise_mul/func.h"
#include "common.h"


int main()
{
    const int N = (int)1e8;
    const float alpha = 1;
    float *host_A, *host_B;
    
    elementwise_mul_param_t param;
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

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    launch_elementwise_mul(param);
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