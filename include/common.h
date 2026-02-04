#ifndef COMMON_H
#define COMMON_H

#include <driver_types.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                          \
{                                                                                 \
    const cudaError_t err = call;                                                 \
    if (err != cudaSuccess)                                                       \
    {                                                                             \
        fprintf(stderr, "CUDA Error at %s:%d - Code: %d, Msg: %s\n",              \
                __FILE__, __LINE__, err, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
}

#define CUBLAS_CHECK(status)                                                      \
{                                                                                 \
    if (status != CUBLAS_STATUS_SUCCESS)                                          \
    {                                                                             \
        fprintf(stderr, "cuBLAS Error at %s:%d - Code: %d\n",                     \
                __FILE__, __LINE__, status);                                      \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
}

#define CUDNN_CHECK(status)                                                       \
{                                                                                 \
    if (status != CUDNN_STATUS_SUCCESS)                                           \
    {                                                                             \
        fprintf(stderr, "cuDNN Error at %s:%d - Code: %d\n",                      \
                __FILE__, __LINE__, status);                                      \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
}



template<typename _Dtype>
int check_result(int N, _Dtype* src, _Dtype* tgt, float atol=1e-5, float rtol=1e-5){
    printf("===================start verfiy===================\n");

    int error=0;
    for(int i=0;i<N; i++){
        bool has_nan = isnan(src[i]) || isinf(src[i]);
        _Dtype diff = std::abs(src[i] - tgt[i]);
        _Dtype scale = std::max(std::abs(src[i]), std::abs(tgt[i]));

        if(has_nan || diff > atol + rtol * scale){
            printf("error, postion:%d, src value:%f, tgt value:%f\n", i, src[i], tgt[i]);
            error++;
            break;
        }        
    }

    printf("================finish,error:%d=========================\n",error);
    return error;
}

#endif