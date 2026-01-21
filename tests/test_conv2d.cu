#include "kernels/conv2d.h"
#include "common.h"


void (*launch_func)(conv2d_param_t) = launch_implgemm;
// 选择测试用的启动函数类型

int main(int argc, char**argv){
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);

    int outh = (h - r + 2*p)/u + 1;
    int outw = (w - s + 2*q)/v + 1;

    float *pIn       = (float*)malloc(n*c*h*w*sizeof(float));
    float *pWeight   = (float*)malloc(k*c*r*s*sizeof(float));
    float *pOut      = (float*)malloc(n*k*outh*outw*sizeof(float));
    float *pOut_verify = (float*)malloc(n*k*outh*outw*sizeof(float));

    float *pIn_device,*pWeight_device,*pOut_device;
    cudaMalloc((void**)&pIn_device, n*c*h*w*sizeof(float));
    cudaMalloc((void**)&pWeight_device, k*c*r*s*sizeof(float));
    cudaMalloc((void**)&pOut_device, n*k*outh*outw*sizeof(float));

    
    for(int i = 0; i < n*c*h*w; i++){
        pIn[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < k*c*r*s; i++){
        pWeight[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < n*k*outh*outw; i++){
        pOut[i] = 0.0;
        pOut_verify[i] = 0.0;
    }

    cudaMemcpy(pIn_device, pIn, n*c*h*w*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_device,pWeight,k*c*r*s*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(pOut_device,pOut, n*k*outh*outw*sizeof(float),cudaMemcpyHostToDevice);


    /*****************************step 1*****************************/
    conv2d_param_t param;
    param.in        = pIn_device;        
    param.weight    = pWeight_device;
    param.out       = pOut_device;         
    param.n         = n;                             
    param.c         = c;                             
    param.h         = h;                             
    param.w         = w;                             
    param.k         = k;                             
    param.r         = r;                             
    param.s         = s;                             
    param.u         = u;                             
    param.v         = v;                             
    param.p         = p;                             
    param.q         = q;
    param.Oh = (h - r + 2*p) / u + 1;
    param.Ow = (w - s + 2*q) / v + 1;                               

    /*****************************step 2*****************************/
    int paramSize = sizeof(conv2d_param_t);
    /*******************************warm up and get result************************************/
    
    launch_conv2d_verify(param);
    cudaMemcpy(pOut_verify, pOut_device,  n*k*outh*outw*sizeof(float), cudaMemcpyDeviceToHost);

    launch_func(param);
    cudaMemcpy(pOut, pOut_device,  n*k*outh*outw*sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    
    float time_elapsed = 0.0;
    int iternum = 100;

    for(int i=0; i<iternum; i++){
        launch_func(param);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    cudaDeviceSynchronize();

    printf("param size :%d \noutput size :%d \ntime: %f us \n",  paramSize, n*k*outh*outw,  time_elapsed * 1000 / iternum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(pIn_device);
        cudaFree(pWeight_device);
        cudaFree(pOut_device);
        
        free(pIn);
        free(pWeight);
        free(pOut);
        free(pOut_verify);
        exit(-1);
    }
    
    check_result(n*k*outh*outw, pOut, pOut_verify);

    cudaFree(pIn_device);
    cudaFree(pWeight_device);
    cudaFree(pOut_device);
    
    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_verify);
    return 0;
}