#include "kernels/attention.h"
#include "common.h"

int main(int argc, char** argv)
{
    int b    = atoi(argv[1]);
    int l_q  = atoi(argv[2]);
    int l_kv = atoi(argv[3]);
    int d    = atoi(argv[4]);
    int seed = 42;

    attention_param_t param;
    param.batch  = b;
    param.len_q  = l_q;
    param.len_kv = l_kv;
    param.dim    = d;

    float* q_host = (float*)malloc(sizeof(float)*b*l_q*d);
    float* k_host = (float*)malloc(sizeof(float)*b*l_kv*d);
    float* v_host = (float*)malloc(sizeof(float)*b*l_kv*d);
    float* o_host = (float*)malloc(sizeof(float)*b*l_q*d);
    float* o_host_verify = (float*)malloc(sizeof(float)*b*l_q*d);

    srand(seed);
    for(int i = 0; i < b*l_q*d; i++){
        q_host[i] = (rand()%255)/255.0;
    }
    for(int i = 0; i < b*l_kv*d; i++){
        k_host[i] = (rand()%255)/255.0;
        v_host[i] = (rand()%255)/255.0;
    }

    CHECK(cudaMalloc((void**)&param.q_ptr, sizeof(float)*b*l_q*d));
    CHECK(cudaMalloc((void**)&param.k_ptr, sizeof(float)*b*l_kv*d));
    CHECK(cudaMalloc((void**)&param.v_ptr, sizeof(float)*b*l_kv*d));
    CHECK(cudaMalloc((void**)&param.o_ptr, sizeof(float)*b*l_q*d));

    CHECK(cudaMemcpy(param.q_ptr, q_host, b*l_q*d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.k_ptr, k_host, b*l_kv*d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(param.v_ptr, v_host, b*l_kv*d, cudaMemcpyHostToDevice));
    

    CHECK(cudaDeviceSynchronize())
    CHECK(cudaMemcpy(o_host, param.o_ptr, b*l_q*d, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(param.q_ptr));
    CHECK(cudaFree(param.k_ptr));
    CHECK(cudaFree(param.v_ptr));
    CHECK(cudaFree(param.o_ptr));

    free(q_host);
    free(k_host);
    free(v_host);
    free(o_host);
    free(o_host_verify);
}