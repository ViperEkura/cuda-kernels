struct elementwise_mul_param_t
{
    float *A;
    float *B;
    float  alpha;
    int    p_size;
};

void launch_elementwise_mul_verify(elementwise_mul_param_t param);