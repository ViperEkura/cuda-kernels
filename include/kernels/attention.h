#ifndef ATTENTION_H
#define ATTENTION_H

struct  attention_param_t
{
    float* q_ptr; // [b, l_q, d]
    float* k_ptr; // [b, l_kv, d]
    float* v_ptr; // [b, l_kv, d]
    float* o_ptr; // [b, l_q, d]

    float  scale;
    float  eps;
    int    batch;
    int    dim;
    int    len_q;
    int    len_kv;
};

void launch_sdqa_attention_fwd_native(attention_param_t param);
void laucnh_sdqa_attention_fwd_flash_v1(attention_param_t param);
#endif