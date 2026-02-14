#include <map>
#include <string>
#include "kernels/softmax.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"

using LaunchFunc = void(*)(softmax_param_t);

float calcu_gflops(float size, float ms)
{
    return 4 * size / (1e6 * ms);
}

int main(int argc, char** argv){
    std::map<std::string, LaunchFunc> func_map = {
        {"native", launch_softmax_native},
        {"smem", launch_softmax_smem},
    };
    ArgParser parser(argc, argv);
    std::string func_name = parser.get("launch_func", "smem");
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
    int outer  = atoi(argv[1]);
    int dim    = atoi(argv[2]);
    int inner  = atoi(argv[3]);
    int iternum = atoi(iter_num.c_str());
    int seed   = 42;

    softmax_param_t param;
    param.outer_size = outer;
    param.softmax_size = dim;
    param.inner_size = inner;

    int size = outer * dim * inner;
    float* src = (float*)malloc(size * sizeof(float));
    float* dst = (float*)malloc(size * sizeof(float));
    float* dst_verify = (float*)malloc(size * sizeof(float));

    srand(seed);
    for(int i = 0; i < size; i++)
    {
       src[i] = (rand()%255)/255.0;
    }

    CUDA_CHECK(cudaMalloc((void**)&param.src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&param.dst, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(param.src, src, size * sizeof(float), cudaMemcpyHostToDevice));

    launch_softmax_native(param);
    CUDA_CHECK(cudaMemcpy(dst_verify, param.dst, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   
    float milliseconds = measure_kernel_runtime(launch_func, param, iternum);
    
    CUDA_CHECK(cudaMemcpy(dst, param.dst, size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(size, milliseconds));
    check_result(size, dst, dst_verify);
    
    CUDA_CHECK(cudaFree(param.src));
    CUDA_CHECK(cudaFree(param.dst));

    free(src);
    free(dst);
    free(dst_verify);
}