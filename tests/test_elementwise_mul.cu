#include "kernels/elementwise_mul.h"
#include "common.h"


void (*launch_func)(elementwise_mul_param_t) = launch_elementwise_mul_vector;

int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    float *lhs, *rhs, *dst, *dst_verify;
    elementwise_mul_param_t param;

    param.N = N;

    lhs = (float*)malloc(sizeof(float) * N);
    rhs = (float*)malloc(sizeof(float) * N);
    dst = (float*)malloc(sizeof(float) * N);
    dst_verify = (float*)malloc(sizeof(float) * N);

    CHECK(cudaMalloc((void**)&param.lhs, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&param.rhs, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&param.dst, sizeof(float) * N));
    
    unsigned seed = 42;
    srand(seed);
    for (int i = 0; i < N; ++i) {
        lhs[i] = (rand() % 255) / 256.0f;
        rhs[i] = (rand() % 255) / 256.0f;
    }

    CHECK(cudaMemcpy(param.lhs, lhs, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.rhs, rhs, sizeof(float) * N, cudaMemcpyHostToDevice));

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    launch_elementwise_mul_vector(param);
    CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CHECK(cudaDeviceSynchronize())
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // 销毁 CUDA 事件
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(dst, param.dst, sizeof(float) * N, cudaMemcpyDeviceToHost));

    launch_func(param);
    CHECK(cudaDeviceSynchronize())
    
    CHECK(cudaMemcpy(dst_verify, param.dst, sizeof(float) * N, cudaMemcpyDeviceToHost));

    printf("===================start verfiy===================\n");

    int error=0;
    for(int i=0;i<N; i++){
        if((fabs(dst[i] - dst_verify[i])) > 0.01 * dst[i] || isnan(dst[i]) || isinf(dst[i])){
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, dst[i], dst_verify[i]);
            error++;
            break;
        }        
    }
    printf("================finish,error:%d=========================\n",error);

    CHECK(cudaFree(param.lhs));
    CHECK(cudaFree(param.rhs));
    CHECK(cudaFree(param.dst));

    free(lhs);
    free(rhs);
    free(dst);
    free(dst_verify);

    return 0;
}