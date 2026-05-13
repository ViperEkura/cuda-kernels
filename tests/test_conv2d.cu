#include "kernels/conv2d.h"
#include "memory.h"
#include "harness.h"
#include "utils/timer.cuh"
#include "common.h"

float calcu_gflops(conv2d_param_t param, float ms)
{
    int oh = (param.h - param.r + 2 * param.p) / param.u + 1;
    int ow = (param.w - param.s + 2 * param.q) / param.v + 1;
    return 2.0 * param.n * param.k * param.c * param.r * param.s * oh * ow / (ms * 1e6);
}

int main(int argc, char** argv)
{
    TestContext<conv2d_param_t> ctx(argc, argv, "implgemm");

    const auto& pos = ctx.parser.positionals();
    if (pos.size() != 11) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  n    Batch size\n");
        fprintf(stderr, "  c    Input channels\n");
        fprintf(stderr, "  h    Input height\n");
        fprintf(stderr, "  w    Input width\n");
        fprintf(stderr, "  k    Output channels (filters)\n");
        fprintf(stderr, "  r    Filter height\n");
        fprintf(stderr, "  s    Filter width\n");
        fprintf(stderr, "  u    Vertical stride\n");
        fprintf(stderr, "  v    Horizontal stride\n");
        fprintf(stderr, "  p    Vertical padding\n");
        fprintf(stderr, "  q    Horizontal padding\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        return EXIT_FAILURE;
    }

    int n = atoi(pos[0].c_str());
    int c = atoi(pos[1].c_str());
    int h = atoi(pos[2].c_str());
    int w = atoi(pos[3].c_str());
    int k = atoi(pos[4].c_str());
    int r = atoi(pos[5].c_str());
    int s = atoi(pos[6].c_str());
    int u = atoi(pos[7].c_str());
    int v = atoi(pos[8].c_str());
    int p = atoi(pos[9].c_str());
    int q = atoi(pos[10].c_str());

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;

    conv2d_param_t param;
    param.n  = n;  param.c  = c;  param.h  = h;  param.w  = w;
    param.k  = k;  param.r  = r;  param.s  = s;
    param.u  = u;  param.v  = v;  param.p  = p;  param.q  = q;
    param.Oh = outh;
    param.Ow = outw;

    HostPtr<float> pIn, pWeight, pOut, pOut_verify;
    pIn.alloc(n * c * h * w);
    pWeight.alloc(k * c * r * s);
    pOut.alloc(n * k * outh * outw);
    pOut_verify.alloc(n * k * outh * outw);

    DevicePtr<float> d_In, d_Weight, d_Out;
    d_In.alloc(n * c * h * w);
    d_Weight.alloc(k * c * r * s);
    d_Out.alloc(n * k * outh * outw);

    rand_fill_255(pIn, n * c * h * w);
    rand_fill_255(pWeight, k * c * r * s);

    param.in     = d_In;
    param.weight = d_Weight;
    param.out    = d_Out;

    CUDA_CHECK(cudaMemcpy(d_In, pIn, sizeof(float) * n * c * h * w, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Weight, pWeight, sizeof(float) * k * c * r * s, cudaMemcpyHostToDevice));

    launch_conv2d_native(param);
    CUDA_CHECK(cudaMemcpy(pOut_verify, d_Out, sizeof(float) * n * k * outh * outw, cudaMemcpyDeviceToHost));

    float ms = measure_kernel_runtime(ctx.launch_func, param, ctx.iternum);

    CUDA_CHECK(cudaMemcpy(pOut, d_Out, sizeof(float) * n * k * outh * outw, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(param, ms));
    check_result(n * k * outh * outw, (float*)pOut, (float*)pOut_verify);

    return 0;
}
