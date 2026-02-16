#include <map>
#include <string>
#include "kernels/conv2d.h"
#include "utils/timer.cuh"
#include "parser.h"
#include "common.h"

using LaunchFunc = void(*)(conv2d_param_t);

float calcu_gflops(conv2d_param_t param, float ms)
{
    int oh = (param.h - param.r + 2 * param.p) / param.u + 1;
    int ow = (param.w - param.s + 2 * param.q) / param.v + 1;

    float total_flops = 2.0 * param.n * param.k * param.c * param.r * param.s * oh * ow;
    return total_flops / (ms * 1e6);
}

int main(int argc, char**argv){

    std::map<std::string, LaunchFunc> func_map = {
        {"native", launch_conv2d_native},
        {"implgemm", launch_implgemm},
        {"winograd", launch_winograd},
    };

    ArgParser parser(argc, argv);
    std::string func_name = parser.get("launch_func", "implgemm");
    std::string iter_num = parser.get("iter", "10");
    
    LaunchFunc launch_func = nullptr;
    auto it = func_map.find(func_name);
    if (it == func_map.end()) {
        fprintf(stderr, "Error: Unknown kernel '%s'. Available kernels: ", func_name.c_str());
        for (const auto& pair : func_map) {
            fprintf(stderr, "%s ", pair.first.c_str());
        }
        fprintf(stderr, "\n");
        return EXIT_FAILURE;
    }

    launch_func = it->second;

    const auto& pos = parser.positionals();
    if (pos.size() != 11) {
        fprintf(stderr, "\nParameters:\n");
        fprintf(stderr, "  n    Batch size\n");
        fprintf(stderr, "  c    Input channels\n");
        fprintf(stderr, "  h    Input height\n");
        fprintf(stderr, "  w    Input width\n");
        fprintf(stderr, "  k    Output channels (filters)\n");
        fprintf(stderr, "  r    Filter height\n");
        fprintf(stderr, "  s    Filter width\n");
        fprintf(stderr, "  u    Vertical stride\n");
        fprintf(stderr, "  v    Horizontal stride\n");
        fprintf(stderr, "  p    Vertical padding\n");
        fprintf(stderr, "  q    Horizontal padding\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --launch_func=NAME\n");
        fprintf(stderr, "  --iter=ITER\n");
    }

    int n = atoi(pos[0].c_str());
    int c = atoi(pos[1].c_str());
    int h = atoi(pos[2].c_str());
    int w = atoi(pos[3].c_str());
    int k = atoi(pos[4].c_str());
    int r = atoi(pos[5].c_str());
    int s = atoi(pos[6].c_str());
    int u = atoi(pos[7].c_str());
    int v = atoi(pos[8].c_str());
    int p = atoi(pos[9].c_str());
    int q = atoi(pos[10].c_str());

    int outh = (h - r + 2*p)/u + 1;
    int outw = (w - s + 2*q)/v + 1;
    int iternum = atoi(iter_num.c_str());

    conv2d_param_t param;      
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

    float *pIn       = (float*)malloc(n*c*h*w*sizeof(float));
    float *pWeight   = (float*)malloc(k*c*r*s*sizeof(float));
    float *pOut      = (float*)malloc(n*k*outh*outw*sizeof(float));
    float *pOut_verify = (float*)malloc(n*k*outh*outw*sizeof(float));

    float *pIn_device,*pWeight_device,*pOut_device;
    CUDA_CHECK(cudaMalloc((void**)&pIn_device, n*c*h*w*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&pWeight_device, k*c*r*s*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&pOut_device, n*k*outh*outw*sizeof(float)));

    param.in        = pIn_device;        
    param.weight    = pWeight_device;
    param.out       = pOut_device;   
    
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

    CUDA_CHECK(cudaMemcpy(pIn_device, pIn, n*c*h*w*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pWeight_device,pWeight,k*c*r*s*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pOut_device,pOut, n*k*outh*outw*sizeof(float),cudaMemcpyHostToDevice));
    
    launch_conv2d_native(param);
    CUDA_CHECK(cudaMemcpy(pOut_verify, pOut_device,  n*k*outh*outw*sizeof(float), cudaMemcpyDeviceToHost));

    float milliseconds = measure_kernel_runtime(launch_func, param, iternum);

    CUDA_CHECK(cudaMemcpy(pOut, pOut_device,  n*k*outh*outw*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Kernel execution speed: %.3f GFLOPS\n", calcu_gflops(param, milliseconds));
    check_result(n*k*outh*outw, pOut, pOut_verify);

    CUDA_CHECK(cudaFree(pIn_device));
    CUDA_CHECK(cudaFree(pWeight_device));
    CUDA_CHECK(cudaFree(pOut_device));
    
    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_verify);
    return 0;
}