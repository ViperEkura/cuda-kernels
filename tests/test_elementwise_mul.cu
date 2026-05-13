#include "kernels/elementwise_mul.h"
#include "memory.h"
#include "harness.h"
#include "utils/timer.cuh"
#include "common.h"

float calcu_gflops(float n, float ms)
{
    return n / (ms * 1e6);
}

int main(int argc, char** argv)
{
    TestContext<elementwise_mul_param_t> ctx(argc, argv, "vector");

    const auto& pos = ctx.parser.positionals();
    if (pos.size() != 1) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  N    Number of elements\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }

    int N = atoi(pos[0].c_str());

    elementwise_mul_param_t param;
    param.N = N;

    HostPtr<float> lhs, rhs, dst, dst_verify;
    lhs.alloc(N);
    rhs.alloc(N);
    dst.alloc(N);
    dst_verify.alloc(N);

    DevicePtr<float> d_lhs, d_rhs, d_dst;
    d_lhs.alloc(N);
    d_rhs.alloc(N);
    d_dst.alloc(N);

    rand_fill(lhs, N);
    rand_fill(rhs, N);

    param.lhs = d_lhs;
    param.rhs = d_rhs;
    param.dst = d_dst;

    CUDA_CHECK(cudaMemcpy(d_lhs, lhs, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs, sizeof(float) * N, cudaMemcpyHostToDevice));

    launch_elementwise_mul_native(param);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dst_verify, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost));

    float ms = measure_kernel_runtime(ctx.launch_func, param, ctx.iternum);

    CUDA_CHECK(cudaMemcpy(dst, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(N, ms));
    check_result(N, (float*)dst, (float*)dst_verify);

    return 0;
}
