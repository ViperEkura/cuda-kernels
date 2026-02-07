#ifndef MATMUL_H
#define MATMUL_H

struct  matmul_param_t
{
    float* lhs; //(M , K)
    float* rhs; //(K , N)
    float* dst;

    int    M;
    int    N;
    int    K;

};

void launch_matmul_native(matmul_param_t param);
void launch_matmul_tiled_v1(matmul_param_t param);
void launch_matmul_tiled_v2(matmul_param_t param);
void launch_matmul_tiled_dbuf(matmul_param_t param);

void launch_matmul_mma(matmul_param_t param);
void launch_matmul_cublas(matmul_param_t param);
#endif