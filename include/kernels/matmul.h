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

void launch_matmul_verify(matmul_param_t param);

#endif