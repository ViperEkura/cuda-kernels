#include "kernels/attention.h"
#include "common.h"

void (*launch_func)(attention_param_t) = launch_sdqa_attention_fwd_native;

float calcu_gflops(float b, float l_q, float l_kv, float d, float ms)
{
    float total_flops = 0;
    // 1. QK^T
    total_flops += b * l_q * l_kv * (2 * d - 1); 
    // 2. Scale
    total_flops += b * l_q * l_kv;
    // 3. Softmax
    total_flops += b * l_q * (5 * l_kv - 2);  
    // 4. Attention·V  
    total_flops += b * l_q * d * (2 * l_kv - 1); 

    //total_flops * 1000 / ms FLOPS;
    return total_flops / (ms * 1e6);
}

int main(int argc, char** argv)
{
    int b    = atoi(argv[1]);
    int l_q  = atoi(argv[2]);
    int l_kv = atoi(argv[3]);
    int d    = atoi(argv[4]);
    int seed = 42;

    attention_param_t param;
    param.batch  = b;
    param.len_q  = l_q;
    param.len_kv = l_kv;
    param.dim    = d;
    param.eps    = 1e-5;
    param.scale  = sqrt(l_kv);

    float* q_host = (float*)malloc(sizeof(float)*b*l_q*d);
    float* k_host = (float*)malloc(sizeof(float)*b*l_kv*d);
    float* v_host = (float*)malloc(sizeof(float)*b*l_kv*d);
    float* o_host = (float*)malloc(sizeof(float)*b*l_q*d);
    float* o_host_verify = (float*)malloc(sizeof(float)*b*l_q*d);

    srand(seed);
    for(int i = 0; i < b*l_q*d; i++){
        q_host[i] = (rand()%255)/255.0;
    }
    for(int i = 0; i < b*l_kv*d; i++){
        k_host[i] = (rand()%255)/255.0;
        v_host[i] = (rand()%255)/255.0;
    }

    CHECK(cudaMalloc((void**)&param.q_ptr, sizeof(float)*b*l_q*d));
    CHECK(cudaMalloc((void**)&param.k_ptr, sizeof(float)*b*l_kv*d));
    CHECK(cudaMalloc((void**)&param.v_ptr, sizeof(float)*b*l_kv*d));
    CHECK(cudaMalloc((void**)&param.o_ptr, sizeof(float)*b*l_q*d));

    CHECK(cudaMemcpy(param.q_ptr, q_host, b*l_q*d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.k_ptr, k_host, b*l_kv*d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.v_ptr, v_host, b*l_kv*d, cudaMemcpyHostToDevice));

    launch_sdqa_attention_fwd_cublas(param);
    CHECK(cudaMemcpy(o_host_verify, param.o_ptr, sizeof(float)*b*l_q*d, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    launch_func(param);
    CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("kernel excution speed: %.3f GFLOPS\n", calcu_gflops(b, l_q, l_kv, d, milliseconds));
    CHECK(cudaMemcpy(o_host, param.o_ptr,sizeof(float)*b*l_q*d, cudaMemcpyDeviceToHost));

    check_result(b*l_kv*d, o_host, o_host_verify);

    CHECK(cudaFree(param.q_ptr));
    CHECK(cudaFree(param.k_ptr));
    CHECK(cudaFree(param.v_ptr));
    CHECK(cudaFree(param.o_ptr));

    free(q_host);
    free(k_host);
    free(v_host);
    free(o_host);
    free(o_host_verify);
}