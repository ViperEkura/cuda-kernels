#include <map>
#include <string>
#include "kernels/attention.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"

using LaunchFunc = void(*)(attention_param_t);

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
    std::map<std::string, LaunchFunc> func_map = {
        {"native", launch_sdqa_attention_fwd_native},
        {"cublas", launch_sdqa_attention_fwd_cublas},
        {"flash_v1", launch_sdqa_attention_fwd_flash_v1},
        {"flash_v2", launch_sdqa_attention_fwd_flash_v2},
        {"flash_v3", launch_sdqa_attention_fwd_flash_v3},
    };

    ArgParser parser(argc, argv);
    std::string func_name = parser.get("launch_func", "flash_v2");
    std::string iter_num = parser.get("iter_num", "10");

    LaunchFunc launch_func = nullptr;
    auto it = func_map.find(func_name);
    if (it == func_map.end()) {
        fprintf(stderr, "Error: Unknown kernel '%s'. Available kernels: ", func_name.c_str());
        for (const auto& pair : func_map) {
            fprintf(stderr, "%s ", pair.first.c_str());
        }
        fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }
    launch_func = it->second;

    const auto& pos = parser.positionals();
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
    int iternum = atoi(iter_num.c_str());
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

    CUDA_CHECK(cudaMalloc((void**)&param.q_ptr, sizeof(float)*b*l_q*d));
    CUDA_CHECK(cudaMalloc((void**)&param.k_ptr, sizeof(float)*b*l_kv*d));
    CUDA_CHECK(cudaMalloc((void**)&param.v_ptr, sizeof(float)*b*l_kv*d));
    CUDA_CHECK(cudaMalloc((void**)&param.o_ptr, sizeof(float)*b*l_q*d));

    CUDA_CHECK(cudaMemcpy(param.q_ptr, q_host, b*l_q*d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(param.k_ptr, k_host, b*l_kv*d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(param.v_ptr, v_host, b*l_kv*d, cudaMemcpyHostToDevice));

    launch_sdqa_attention_fwd_cublas(param);
    CUDA_CHECK(cudaMemcpy(o_host_verify, param.o_ptr, sizeof(float)*b*l_q*d, cudaMemcpyDeviceToHost));
   
    float milliseconds = measure_kernel_runtime(launch_func, param, iternum);
    
    CUDA_CHECK(cudaMemcpy(o_host, param.o_ptr,sizeof(float)*b*l_q*d, cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(b, l_q, l_kv, d, milliseconds));
    check_result(b*l_q*d, o_host, o_host_verify, 1e-3);

    CUDA_CHECK(cudaFree(param.q_ptr));
    CUDA_CHECK(cudaFree(param.k_ptr));
    CUDA_CHECK(cudaFree(param.v_ptr));
    CUDA_CHECK(cudaFree(param.o_ptr));

    free(q_host);
    free(k_host);
    free(v_host);
    free(o_host);
    free(o_host_verify);
}