#include "kernels/elementwise_mul.h"
#include "common.h"


void (*launch_func)(elementwise_mul_param_t) = launch_elementwise_mul_vector;

float calcu_gflops(float n, float ms)
{
    return n / (ms * 1e6);
}

int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    float *lhs, *rhs, *dst, *dst_verify;
    elementwise_mul_param_t param;

    param.N = N;

    lhs = (float*)malloc(sizeof(float) * N);
    rhs = (float*)malloc(sizeof(float) * N);
    dst = (float*)malloc(sizeof(float) * N);
    dst_verify = (float*)malloc(sizeof(float) * N);

    CUDA_CHECK(cudaMalloc((void**)&param.lhs, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&param.rhs, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&param.dst, sizeof(float) * N));
    
    unsigned seed = 42;
    srand(seed);
    for (int i = 0; i < N; ++i) {
        lhs[i] = (rand() % 255) / 256.0f;
        rhs[i] = (rand() % 255) / 256.0f;
    }

    CUDA_CHECK(cudaMemcpy(param.lhs, lhs, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(param.rhs, rhs, sizeof(float) * N, cudaMemcpyHostToDevice));

    launch_elementwise_mul_native(param);
    CUDA_CHECK(cudaDeviceSynchronize())
    CUDA_CHECK(cudaMemcpy(dst_verify, param.dst, sizeof(float) * N, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch_elementwise_mul_vector(param);
    CUDA_CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaDeviceSynchronize())
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(dst, param.dst, sizeof(float) * N, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(N, milliseconds));


    check_result(N, dst, dst_verify);    

    CUDA_CHECK(cudaFree(param.lhs));
    CUDA_CHECK(cudaFree(param.rhs));
    CUDA_CHECK(cudaFree(param.dst));

    free(lhs);
    free(rhs);
    free(dst);
    free(dst_verify);

    return 0;
}