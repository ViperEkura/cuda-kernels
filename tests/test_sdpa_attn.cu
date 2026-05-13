#include "kernels/attention.h"
#include "memory.h"
#include "harness.h"
#include "utils/timer.cuh"
#include "common.h"

float calcu_gflops(float b, float l_q, float l_kv, float d, float ms)
{
    float total_flops = 0;
    total_flops += b * l_q * l_kv * (2 * d - 1);
    total_flops += b * l_q * l_kv;
    total_flops += b * l_q * (5 * l_kv - 2);
    total_flops += b * l_q * d * (2 * l_kv - 1);
    return total_flops / (ms * 1e6);
}

int main(int argc, char** argv)
{
    TestContext<attention_param_t> ctx(argc, argv, "flash_v2");

    const auto& pos = ctx.parser.positionals();
    if (pos.size() != 4) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  batch     Batch size\n");
        fprintf(stderr, "  len_q     Query sequence length\n");
        fprintf(stderr, "  len_kv    Key/Value sequence length\n");
        fprintf(stderr, "  dim       Hidden dimension\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
        return EXIT_FAILURE;
    }

    int b    = atoi(pos[0].c_str());
    int l_q  = atoi(pos[1].c_str());
    int l_kv = atoi(pos[2].c_str());
    int d    = atoi(pos[3].c_str());

    attention_param_t param;
    param.batch  = b;
    param.len_q  = l_q;
    param.len_kv = l_kv;
    param.dim    = d;
    param.eps    = 1e-5;
    param.scale  = sqrt(l_kv);

    HostPtr<float> q_host, k_host, v_host, o_host, o_host_verify;
    q_host.alloc(b * l_q * d);
    k_host.alloc(b * l_kv * d);
    v_host.alloc(b * l_kv * d);
    o_host.alloc(b * l_q * d);
    o_host_verify.alloc(b * l_q * d);

    DevicePtr<float> d_Q, d_K, d_V, d_O;
    d_Q.alloc(b * l_q * d);
    d_K.alloc(b * l_kv * d);
    d_V.alloc(b * l_kv * d);
    d_O.alloc(b * l_q * d);

    rand_fill_255(q_host, b * l_q * d);
    rand_fill_255(k_host, b * l_kv * d);
    rand_fill_255(v_host, b * l_kv * d);

    param.q_ptr = d_Q;
    param.k_ptr = d_K;
    param.v_ptr = d_V;
    param.o_ptr = d_O;

    CUDA_CHECK(cudaMemcpy(d_Q, q_host, sizeof(float) * b * l_q * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, k_host, sizeof(float) * b * l_kv * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, v_host, sizeof(float) * b * l_kv * d, cudaMemcpyHostToDevice));

    launch_sdqa_attention_fwd_cublas(param);
    CUDA_CHECK(cudaMemcpy(o_host_verify, d_O, sizeof(float) * b * l_q * d, cudaMemcpyDeviceToHost));

    float ms = measure_kernel_runtime(ctx.launch_func, param, ctx.iternum);

    CUDA_CHECK(cudaMemcpy(o_host, d_O, sizeof(float) * b * l_q * d, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(b, l_q, l_kv, d, ms));
    check_result(b * l_q * d, (float*)o_host, (float*)o_host_verify, 1e-3);

    return 0;
}
