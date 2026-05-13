#include "kernels/softmax.h"
#include "memory.h"
#include "harness.h"
#include "utils/timer.cuh"
#include "common.h"

float calcu_gflops(float size, float ms)
{
    return 4 * size / (1e6 * ms);
}

int main(int argc, char** argv)
{
    TestContext<softmax_param_t> ctx(argc, argv, "smem");

    const auto& pos = ctx.parser.positionals();
    if (pos.size() != 3) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  outer    Outer size (batch size)\n");
        fprintf(stderr, "  dim      Softmax dimension\n");
        fprintf(stderr, "  inner    Inner size\n");
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }

    int outer = atoi(pos[0].c_str());
    int dim   = atoi(pos[1].c_str());
    int inner = atoi(pos[2].c_str());

    softmax_param_t param;
    param.outer_size   = outer;
    param.softmax_size = dim;
    param.inner_size   = inner;

    int size = outer * dim * inner;

    HostPtr<float> src, dst, dst_verify;
    src.alloc(size);
    dst.alloc(size);
    dst_verify.alloc(size);

    DevicePtr<float> d_src, d_dst;
    d_src.alloc(size);
    d_dst.alloc(size);

    rand_fill_255(src, size);

    param.src = d_src;
    param.dst = d_dst;

    CUDA_CHECK(cudaMemcpy(d_src, src, sizeof(float) * size, cudaMemcpyHostToDevice));

    launch_softmax_native(param);
    CUDA_CHECK(cudaMemcpy(dst_verify, d_dst, sizeof(float) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = measure_kernel_runtime(ctx.launch_func, param, ctx.iternum);

    CUDA_CHECK(cudaMemcpy(dst, d_dst, sizeof(float) * size, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(size, ms));
    check_result(size, (float*)dst, (float*)dst_verify);

    return 0;
}
