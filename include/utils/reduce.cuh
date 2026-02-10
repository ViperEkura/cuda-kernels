#ifndef UTILS_REDUCE_CUH
#define UTILS_REDUCE_CUH

// ------------------ Warp-level reductions ------------------
template<typename T>
__forceinline__ __device__ T warpReduceMax(T val, int width=32) {
    for (int offset = width / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset, width));
    return val;
}

template<typename T>
__forceinline__ __device__ T warpReduceMin(T val, int width=32) {
    for (int offset = width / 2; offset > 0; offset /= 2)
        val = min(val, __shfl_xor_sync(0xffffffff, val, offset, width));
    return val;
}

template<typename T>
__forceinline__ __device__ T warpReduceSum(T val, int width=32) {
    for (int offset = width / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset, width);
    return val;
}

// ------------------ Block-level reductions ------------------

#endif
