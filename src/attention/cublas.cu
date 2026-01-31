#include <cublas_v2.h>

#include "kernels/attention.h"
#include "common.h"


__global__ void softmax_kernel(float* input, int rows, int cols, float scale) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int col = tid;

    if (row >= rows) return;
    float* x = input + row * cols;

    // Find max in this row
    float thread_max = -INFINITY;
    for (int i = col; i < cols; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x[i] * scale);
    }

    // Reduce max within block
    shared_data[col] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            shared_data[col] = fmaxf(shared_data[col], shared_data[col + stride]);
        }
        __syncthreads();
    }
    float row_max = shared_data[0];
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = col; i < cols; i += blockDim.x) {
        float val = __expf(x[i] * scale - row_max);
        x[i] = val;
        thread_sum += val;
    }

    shared_data[col] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            shared_data[col] += shared_data[col + stride];
        }
        __syncthreads();
    }
    float row_sum = shared_data[0];
    __syncthreads();

    // Normalize
    float inv_row_sum = 1.0f / (row_sum + 1e-8f);
    for (int i = col; i < cols; i += blockDim.x) {
        x[i] *= inv_row_sum;
    }
}
void launch_sdqa_attention_fwd_cublas(attention_param_t param) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    float* d_scores = nullptr;
    size_t scores_size = param.batch * param.len_q * param.len_kv * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 使用cublasSgemmStridedBatched - 推荐，当矩阵连续存储时
    int stride_q = param.len_q * param.dim;
    int stride_k = param.len_kv * param.dim;
    int stride_v = param.len_kv * param.dim;
    int stride_scores = param.len_q * param.len_kv;
    int stride_output = param.len_q * param.dim;
    
    // 步骤1: 批处理Q * K^T

    CUBLAS_CHECK(cublasSgemmStridedBatched(
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
    
    const int BLOCK_SIZE = 256;
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);

    dim3 grid(param.batch * param.len_q);
    dim3 block(BLOCK_SIZE);

    softmax_kernel<<<grid, block, shared_mem_size>>>(
        d_scores,
        param.batch * param.len_q,
        param.len_kv,
        param.scale
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 步骤3: 批处理softmax(QK^T) * V
    CUBLAS_CHECK(cublasSgemmStridedBatched(
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
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_scores));
}
