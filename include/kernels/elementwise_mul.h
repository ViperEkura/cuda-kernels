#ifndef ELEMENTWISE_MUL
#define ELEMENTWISE_MUL

struct elementwise_mul_param_t
{
    float* lhs;
    float* rhs;
    float* dst;
    int    N;
};

void launch_elementwise_mul_verify(elementwise_mul_param_t param);
void launch_elementwise_mul_vector(elementwise_mul_param_t param);

#endif