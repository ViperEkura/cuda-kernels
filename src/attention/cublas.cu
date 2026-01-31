#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "kernels/attention.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void softmax_kernel(float* input, int rows, int cols, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float* row_ptr = input + row * cols;
    

    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        float val = row_ptr[i] * scale;
        if (val > max_val) max_val = val;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = expf(row_ptr[i] * scale - max_val);
        row_ptr[i] = val;
        sum += val;
    }
    
    sum = 1.0f / (sum + 1e-8f);
    for (int i = 0; i < cols; i++) {
        row_ptr[i] *= sum;
    }
}

void launch_sdqa_attention_fwd_cublas(attention_param_t param) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    float* d_scores = nullptr;
    size_t scores_size = param.batch * param.len_q * param.len_kv * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_scores, scores_size));
    

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 使用cublasSgemmStridedBatched - 推荐，当矩阵连续存储时
    int stride_q = param.len_q * param.dim;
    int stride_k = param.len_kv * param.dim;
    int stride_v = param.len_kv * param.dim;
    int stride_scores = param.len_q * param.len_kv;
    int stride_output = param.len_q * param.dim;
    
    // 步骤1: 批处理Q * K^T

    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,    // K需要转置为K^T
        CUBLAS_OP_N,    // Q不转置
        param.len_kv,   // M: K^T的行数 = len_kv
        param.len_q,    // N: Q的列数 = len_q
        param.dim,      // K: 共同的dim维度
        &alpha,
        param.k_ptr,    // K矩阵
        param.dim,      // lda: K的leading dimension
        stride_k,       // strideA: batch之间的步长
        param.q_ptr,    // Q矩阵
        param.dim,      // ldb: Q的leading dimension
        stride_q,       // strideB: batch之间的步长
        &beta,
        d_scores,       // 输出矩阵
        param.len_kv,   // ldc: scores的leading dimension
        stride_scores,  // strideC: batch之间的步长
        param.batch     // batch数量
    ));
    
    // 步骤2: 缩放并应用softmax
    int threads_per_block = 256;
    int blocks_per_grid = (param.batch * param.len_q + threads_per_block - 1) / threads_per_block;
    
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_scores, 
        param.batch * param.len_q, 
        param.len_kv, 
        param.scale
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 步骤3: 批处理softmax(QK^T) * V
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,    // V不转置
        CUBLAS_OP_N,    // scores不转置
        param.dim,      // M: V的行数 = dim
        param.len_q,    // N: scores的列数 = len_q
        param.len_kv,   // K: 共同的len_kv维度
        &alpha,
        param.v_ptr,    // V矩阵
        param.dim,      // lda: V的leading dimension
        stride_v,       // strideA: batch之间的步长
        d_scores,       // scores矩阵
        param.len_kv,   // ldb: scores的leading dimension
        stride_scores,  // strideB: batch之间的步长
        &beta,
        param.o_ptr,    // 输出矩阵
        param.dim,      // ldc: output的leading dimension
        stride_output,  // strideC: batch之间的步长
        param.batch     // batch数量
    ));
    
    // 清理
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_scores));
}
