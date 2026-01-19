struct  matmul_param_t
{
    float* src_A; //(M , K)
    float* src_B; //(K , N)
    float* dst;

    int    M;
    int    N;
    int    K;

};

void launch_matmul_verify(matmul_param_t param);