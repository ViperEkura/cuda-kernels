#ifndef MEMORY_H
#define MEMORY_H

#include "common.h"

template <typename T>
struct DevicePtr {
    T* ptr = nullptr;

    DevicePtr() = default;

    void alloc(size_t n) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
        CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(T)));
    }

    ~DevicePtr() { if (ptr) cudaFree(ptr); }
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;
    operator T*() const { return ptr; }
};

template <typename T>
struct HostPtr {
    T* ptr = nullptr;

    HostPtr() = default;

    void alloc(size_t n) {
        if (ptr) free(ptr);
        ptr = (T*)malloc(n * sizeof(T));
    }

    ~HostPtr() { if (ptr) free(ptr); }
    HostPtr(const HostPtr&) = delete;
    HostPtr& operator=(const HostPtr&) = delete;
    operator T*() const { return ptr; }
};

#endif
