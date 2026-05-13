#include <string>
#include "kernels/elementwise_mul.h"
#include "registry.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"


float calcu_gflops(float n, float ms)
{
    return n / (ms * 1e6);
}

int main(int argc, char** argv)
{
    ArgParser parser(argc, argv);
    std::string func_name = parser.get("launch_func", "vector");
    std::string iter_num = parser.get("iter", "10");

    auto launch_func = KernelRegistry<elementwise_mul_param_t>::lookup(func_name);

    const auto& pos = parser.positionals();
    if (pos.size() != 1) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  N    Number of elements\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        fprintf(stderr, "\n");
        return 1;
    }

    int N = atoi(pos[0].c_str());
    int iternum = atoi(iter_num.c_str());

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

    float milliseconds = measure_kernel_runtime(launch_func, param, iternum);

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