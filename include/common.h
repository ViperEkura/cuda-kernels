#ifndef COMMON_H
#define COMMON_H

#include <driver_types.h>
#include <stdio.h>

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("Error: %s, Line: %d\n", __FILE__, __LINE__);                    \
        printf("Error code: %d, Reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-1);                                                               \
    }                                                                           \
}

#endif