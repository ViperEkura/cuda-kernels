#ifndef CONV2D_H
#define CONV2D_H

// use nchw layout
struct conv2d_param_t {
    float* in;               // Input data address 
    float* weight;           // Weight data address
    float* out;              // Output data address
    unsigned int n;          // Batch size
    unsigned int c;          // Number of input channels
    unsigned int h;          // Input height
    unsigned int w;          // Input width
    unsigned int k;          // Number of filters (output channels)
    unsigned int r;          // Filter height
    unsigned int s;          // Filter width
    unsigned int u;          // Vertical stride (height direction)
    unsigned int v;          // Horizontal stride (width direction)
    unsigned int p;          // Vertical padding (height direction)
    unsigned int q;          // Horizontal padding (width direction)
    unsigned int Oh;         // Output height
    unsigned int Ow;         // Output width
};

void launch_conv2d_native(conv2d_param_t param);
void launch_implgemm(conv2d_param_t param);
void launch_winograd(conv2d_param_t param);

#endif
