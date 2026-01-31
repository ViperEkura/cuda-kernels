#include <cublas_v2.h>
#include <stdexcept>

#include "kernels/matmul.h"

void launch_matmul_cublas(matmul_param_t param) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS handle creation failed");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cublasSgemm(handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         param.M,
                         param.N,
                         param.K,
                         &alpha,
                         param.rhs,
                         param.N, 
                         param.lhs,
                         param.K,
                         &beta,
                         param.dst,
                         param.N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS SGEMM failed");
    }

    cudaDeviceSynchronize();
    cublasDestroy(handle);
}
