#ifndef HARNESS_H
#define HARNESS_H

#include <cstdlib>
#include <cstdio>
#include "parser.h"
#include "registry.h"

inline void rand_fill(float* p, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; ++i)
        p[i] = (rand() % 255) / 256.0f;
}

inline void rand_fill_255(float* p, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; ++i)
        p[i] = (rand() % 255) / 255.0f;
}

template <typename ParamType>
struct TestContext {
    ArgParser parser;
    int iternum;
    void (*launch_func)(ParamType);

    TestContext(int argc, char** argv, const char* default_kernel)
        : parser(argc, argv)
    {
        iternum = atoi(parser.get("iter", "10").c_str());
        launch_func = KernelRegistry<ParamType>::lookup(
            parser.get("launch_func", default_kernel));
    }
};

#endif
