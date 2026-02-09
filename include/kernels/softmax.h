#ifndef SOFTMAX_H
#define SOFTMAX_H

struct softmax_param_t
{
    float* src;
    float* dst;
    int    total_size;
    int    softmax_size;
    int    softmax_stride;
};

/*
    [N, C, H, W], reduce on C
    softmax_size = C
    softmax_stride = H * W
    total_size = N * C * H * W

    [N, d] reduce on d
    softmax_size = d
    softmax_stride = 1
    total_size = N * d

    chunks = total_size / softmax_stride
*/

void launch_softmax_native(softmax_param_t param);

#endif