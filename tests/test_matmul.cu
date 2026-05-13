#include "kernels/matmul.h"
#include "memory.h"
#include "harness.h"
#include "utils/timer.cuh"
#include "common.h"

float calcu_gflops(float m, float n, float k, float ms)
{
    return 2 * m * n * k / (ms * 1e6);
}

int main(int argc, char** argv)
{
    TestContext<matmul_param_t> ctx(argc, argv, "tiled_v3");

    const auto& pos = ctx.parser.positionals();
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

    matmul_param_t param;
    param.M = M;
    param.N = N;
    param.K = K;

    HostPtr<float> host_A, host_B, host_C, host_C_verify;
    host_A.alloc(M * K);
    host_B.alloc(N * K);
    host_C.alloc(M * N);
    host_C_verify.alloc(M * N);

    DevicePtr<float> d_A, d_B, d_C;
    d_A.alloc(M * K);
    d_B.alloc(N * K);
    d_C.alloc(M * N);

    rand_fill(host_A, M * K);
    rand_fill(host_B, N * K);

    param.lhs = d_A;
    param.rhs = d_B;
    param.dst = d_C;

    CUDA_CHECK(cudaMemcpy(d_A, host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice));

    launch_matmul_cublas(param);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_C_verify, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    float ms = measure_kernel_runtime(ctx.launch_func, param, ctx.iternum);

    CUDA_CHECK(cudaMemcpy(host_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(M, N, K, ms));
    check_result(M * N, (float*)host_C, (float*)host_C_verify, 5e-5, 2e-5);

    return 0;
}
