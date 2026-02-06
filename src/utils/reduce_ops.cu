#include "utils/reduce.cuh"

__forceinline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ float warpReduceMin(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
