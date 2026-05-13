#include "kernels/matmul.h"
#include "registry.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"

float calcu_gflops(float m, float n, float k, float ms)
{
    float total_flops = 2 * m * n * k;
    //total_flops * 1000 / ms FLOPS;
    return total_flops / (ms * 1e6);
}

int main(int argc, char** argv)
{
    ArgParser parser = ArgParser(argc, argv);
    std::string func = parser.get("launch_func", "tiled_v3");
    std::string iter_num = parser.get("iter", "10");
    auto launch_func = KernelRegistry<matmul_param_t>::lookup(func);

    const auto& pos = parser.positionals();
    if (pos.size() != 3) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  m    First matrix rows (M)\n");
        fprintf(stderr, "  n    Second matrix columns (N)\n");
        fprintf(stderr, "  k    Inner dimension (K)\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        return EXIT_FAILURE;
    }

    int M = atoi(pos[0].c_str());
    int N = atoi(pos[1].c_str());
    int K = atoi(pos[2].c_str());
    int iternum = atoi(iter_num.c_str());
    
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

    float milliseconds = measure_kernel_runtime(launch_func, param, iternum);

    CUDA_CHECK(cudaMemcpy(host_C, param.dst, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(M, N, K, milliseconds));
    check_result(N, host_C, host_C_verify, 5e-5, 2e-5); //  for mixed precision 

    CUDA_CHECK(cudaFree(param.lhs));
    CUDA_CHECK(cudaFree(param.rhs));
    CUDA_CHECK(cudaFree(param.dst));

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_C_verify);

    return 0;
}