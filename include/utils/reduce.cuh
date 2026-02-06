#ifndef UTILS_REDUCE_CUH
#define UTILS_REDUCE_CUH

__forceinline__ __device__ float warpReduceMax(float val);
__forceinline__ __device__ float warpReduceMin(float val);
__forceinline__ __device__ float warpReduceSum(float val);

#endif
