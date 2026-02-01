#ifndef SOFTMAX_H
#define SOFTMAX_H

struct softmax_param_t
{
    float* src;
    float* dst;
    int    size;
    int    stride;
};

void launch_softmax_native(softmax_param_t param);

#endif