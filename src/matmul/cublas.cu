#include <cublas_v2.h>
#include "kernels/matmul.h"
#include "common.h"

void launch_matmul_cublas(matmul_param_t param) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // (col major) C^T(N×M) = (col major) A^T(N×K) * (col major) B^T(K×M)
    // (row major) C (M, N) = (col major) C^T(N×M)
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