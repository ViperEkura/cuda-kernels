#include "kernels/softmax.h"
#include "common.h"

void (*launch_func)(softmax_param_t) = launch_softmax_native;

float calcu_gflops(float size, float ms)
{
    return 4 * size / (1e6 * ms);
}

int main(int argc, char** argv){
    int size   = atoi(argv[1]);
    int stride = atoi(argv[2]);
    int seed   = 42;

    softmax_param_t param;
    param.size   = size;
    param.stride = stride;

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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch_func(param);
    CUDA_CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(dst, param.dst, size * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(size, milliseconds));

    check_result(size, dst, dst_verify);
    
    cudaFree(param.src);
    cudaFree(param.dst);

    free(src);
    free(dst);
    free(dst_verify);
}