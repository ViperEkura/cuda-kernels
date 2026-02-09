#include "kernels/softmax.h"
#include "utils/reduce.cuh"
#include <cstdio>

static constexpr int Bd = 256;
static constexpr int WARP_SIZE = 32;

__global__ void softmax_smem(softmax_param_t param)
{
    extern __shared__ float sdata[]; 
    __shared__ float reduce[Bd / WARP_SIZE];

    int chunks = param.outer_size * param.inner_size;
    int chunk_id = blockIdx.x;
    if (chunk_id >= chunks) return;

    int wrap_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int outer_id = chunk_id / param.inner_size;
    int inner_id = chunk_id % param.inner_size;
    int offset = outer_id * param.inner_size * param.softmax_size + inner_id;
    
    float thread_max = -INFINITY;
    float global_max = -INFINITY;
    float thread_sum = 0;
    float global_sum = 0;

    // find_max
    for (int i =threadIdx.x; i < param.softmax_size; i += blockDim.x)
    {
        int idx = offset + i * param.inner_size;
        // to smem
        sdata[i] = param.src[idx];
        thread_max = max(thread_max, sdata[i]);
    }
    
    thread_max = warpReduceMax(thread_max);
    if (lane_id == 0) reduce[wrap_id] = thread_max;
    __syncthreads();

    if (wrap_id == 0)
    {
        thread_max = (threadIdx.x < Bd / WARP_SIZE) ? reduce[threadIdx.x] : - INFINITY;
        thread_max = warpReduceMax(thread_max);
        if (lane_id == 0)
        {
            reduce[0] = thread_max;
        }
        
    }
    __syncthreads();
    global_max = reduce[0];

    // calcu exp
    for (int i = threadIdx.x; i < param.softmax_size; i += blockDim.x)
    {
        thread_sum += exp(sdata[i] - global_max);
    }

    thread_sum = warpReduceSum(thread_sum);
    if (lane_id == 0) reduce[wrap_id] = thread_sum;
    __syncthreads();

    if (wrap_id == 0)
    {
        thread_sum = (threadIdx.x < Bd / WARP_SIZE) ? reduce[threadIdx.x] : 0;
        thread_sum = warpReduceSum(thread_sum);
        if (lane_id == 0)
        {
            reduce[0] = thread_sum;
        }
    }
    __syncthreads();
    global_sum = reduce[0];

    // write back
    float inv_sum = 1 / global_sum;

    for (int i = threadIdx.x; i < param.softmax_size; i += blockDim.x)
    {
        int idx = offset + i * param.inner_size;
        param.dst[idx] = exp(sdata[i] - global_max) * inv_sum;
    }
}

void launch_softmax_smem(softmax_param_t param)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threads = Bd;
    int chunks = param.outer_size * param.inner_size;
    int blocks = (chunks + Bd - 1) / Bd;
    int smem_size = param.softmax_size * sizeof(float); 
    int warp_reduce_smem = (threads / WARP_SIZE) * sizeof(float);
    int total_smem = smem_size + warp_reduce_smem;
    
    if (total_smem > prop.sharedMemPerBlock) {
        fprintf(stderr, "ERROR: Required shared memory %d bytes exceeds device limit %zu bytes\n", 
                total_smem, prop.sharedMemPerBlock);
        return;
    }
    
    softmax_smem<<<blocks, threads, total_smem>>>(param);
}