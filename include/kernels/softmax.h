#ifndef SOFTMAX_H
#define SOFTMAX_H

struct softmax_param_t
{
    float* src;
    float* dst;
    int    outer_size;      // number of elements before the dim dimension
    int    softmax_size;    // length of the softmax dimension
    int    inner_size;      // number of elements after the dim dimension
};
// total_size = outer_size * softmax_size * inner_size


void launch_softmax_native(softmax_param_t param);
void launch_softmax_smem(softmax_param_t param);

#endif