#ifndef UTILS_OPS_H
#define UTILS_OPS_H

template<typename T>
__host__ __device__ constexpr T MIN(const T& a, const T& b) 
{
    return (a < b) ? a : b;
}

template<typename T>
__host__ __device__ constexpr T MAX(const T& a, const T& b) 
{
    return (a < b) ? a : b;
}


template<typename T>
__host__ __device__ constexpr T CDIV(const T& a, const T& b)
{
    return (a + b - 1) / b;
}

#endif

