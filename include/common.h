#ifndef COMMON_H
#define COMMON_H

#include <driver_types.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <map>
#include <string>

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

#define LAUNCH_CHECK()                                                            \
{                                                                                 \
    const cudaError_t err = cudaPeekAtLastError();                                \
    if (err != cudaSuccess)                                                       \
    {                                                                             \
        fprintf(stderr, "Kernel Launch Error at %s:%d - Code: %d, Msg: %s\n",     \
                __FILE__, __LINE__, err, cudaGetErrorString(err));                \
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

template<typename LaunchFunc>
LaunchFunc lookup_kernel(const std::map<std::string, LaunchFunc>& func_map,
                          const std::string& name) {
    auto it = func_map.find(name);
    if (it != func_map.end()) return it->second;
    fprintf(stderr, "Error: Unknown kernel '%s'. Available kernels: ", name.c_str());
    for (const auto& pair : func_map) fprintf(stderr, "%s ", pair.first.c_str());
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

#endif