#ifndef CONV2D_DEFINES_H
#define CONV2D_DEFINES_H

struct param_t{
    float*   in;                             //输入数据地址
    float*   weight;                         //权值数据地址
    float*   out;                            //输出数据地址
    unsigned int      n;                              //batch size
    unsigned int      c;                              //channel number
    unsigned int      h;                              //数据高
    unsigned int      w;                              //数据宽
    unsigned int      k;                              //卷积核数量
    unsigned int      r;                              //卷积核高
    unsigned int      s;                              //卷积核宽 
    unsigned int      u;                              //卷积在高方向上的步长
    unsigned int      v;                              //卷积在宽方向上的步长
    unsigned int      p;                              //卷积在高方向上的补边
    unsigned int      q;                              //卷积在宽方向上的补边
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
};

#endif