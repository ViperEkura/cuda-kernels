#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include "common.h"

template<typename _Func, typename _Param>
float measure_kernel_runtime(_Func launch_func, _Param param, int iternum)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < iternum; i++){
        launch_func(param);
    }
    CUDA_CHECK(cudaEventRecord(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds / iternum;
}

#endif

