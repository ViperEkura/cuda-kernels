#ifndef UTILS_REDUCE_CUH
#define UTILS_REDUCE_CUH

// ------------------ Warp-level reductions ------------------
template<typename T>
__forceinline__ __device__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename T>
__forceinline__ __device__ T warpReduceMin(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ------------------ Block-level reductions ------------------

#endif
