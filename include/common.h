#ifndef COMMON_H
#define COMMON_H

#include <driver_types.h>
#include <stdio.h>

#define CHECK(call)                                                               \
{                                                                                 \
    const cudaError_t error = call;                                               \
    if (error != cudaSuccess)                                                     \
    {                                                                             \
        printf("Error: %s, Line: %d\n", __FILE__, __LINE__);                      \
        printf("Error code: %d, Reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-1);                                                                 \
    }                                                                             \
}

int check_kernel_launch()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<typename _Dtype>
int check_result(int N, _Dtype* src, _Dtype* tgt){
    printf("===================start verfiy===================\n");

    int error=0;
    for(int i=0;i<N; i++){
        if((fabs(src[i] - tgt[i])) > 0.01 * tgt[i] || isnan(src[i]) || isinf(src[i])){
            printf("error, postion:%d, src value:%f, tgt value:%f\n", i, src[i], tgt[i]);
            error++;
            break;
        }        
    }

    printf("================finish,error:%d=========================\n",error);
    return error;
}

#endif