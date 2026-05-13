#include <string>
#include "kernels/softmax.h"
#include "registry.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"

float calcu_gflops(float size, float ms)
{
    return 4 * size / (1e6 * ms);
}

int main(int argc, char** argv){
    ArgParser parser(argc, argv);
    std::string func_name = parser.get("launch_func", "smem");
    std::string iter_num = parser.get("iter", "10");
    auto launch_func = KernelRegistry<softmax_param_t>::lookup(func_name);

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
    int outer  = atoi(pos[0].c_str());
    int dim    = atoi(pos[1].c_str());
    int inner  = atoi(pos[2].c_str());
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