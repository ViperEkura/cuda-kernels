#include <cublas_v2.h>
#include "kernels/matmul.h"
#include "common.h"

void launch_matmul_cublas(matmul_param_t param) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        param.N,
        param.M,
        param.K,
        &alpha,
        param.rhs,
        param.N, 
        param.lhs,
        param.K,
        &beta,
        param.dst,
        param.N
    ));

    cudaDeviceSynchronize();
    cublasDestroy(handle);
}