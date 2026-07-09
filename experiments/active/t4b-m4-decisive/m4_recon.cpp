/* [G3-cert-hardening M4] gguf tensor inventory + dmin!=0 scan.
 * Reads DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf metadata (no_alloc), lists every Q4_K/Q5_K/Q6_K
 * tensor with its shape, then for each Q4_K/Q5_K tensor whose shape fits the 16x1 repack-GEMM
 * (ne[0]%256==0 && ne[1]%16==0) reads the first N super-blocks straight from the file and counts
 * how many have dmin!=0 -- so M4 can pick a REAL weight tensor with an ACTIVE min-term. Read-only. */
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

struct blk_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };   /* 144 */
struct blk_q5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }; /* 176 */

static float h2f(uint16_t h){
    uint32_t s=(h>>15)&1u,e=(h>>10)&0x1Fu,m=h&0x3FFu,o;
    if(e==0){ if(m==0)o=s<<31; else { int ee=-14; while(!(m&0x400u)){m<<=1;ee--;} m&=0x3FFu; o=(s<<31)|((uint32_t)(ee+127)<<23)|(m<<13);} }
    else if(e==0x1Fu)o=(s<<31)|(0xFFu<<23)|(m<<13);
    else o=(s<<31)|((e-15u+127u)<<23)|(m<<13);
    float f; memcpy(&f,&o,4); return f;
}

int main(int argc,char**argv){
    const char* path = argc>1?argv[1]:"/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";
    struct ggml_context* mctx=nullptr;
    struct gguf_init_params p{ true, &mctx };
    struct gguf_context* g = gguf_init_from_file(path, p);
    if(!g){ printf("FAILED to open %s\n", path); return 1; }
    int64_t nt = gguf_get_n_tensors(g);
    size_t data_off = gguf_get_data_offset(g);
    printf("model=%s  n_tensors=%lld  data_off=%zu\n", path, (long long)nt, data_off);
    FILE* f = fopen(path,"rb");
    for(int64_t i=0;i<nt;i++){
        const char* name = gguf_get_tensor_name(g,i);
        enum ggml_type ty = gguf_get_tensor_type(g,i);
        struct ggml_tensor* t = ggml_get_tensor(mctx, name);
        int64_t ne0=t->ne[0], ne1=t->ne[1], ne2=t->ne[2], ne3=t->ne[3];
        const char* tn = (ty==GGML_TYPE_Q4_K)?"Q4_K":(ty==GGML_TYPE_Q5_K)?"Q5_K":(ty==GGML_TYPE_Q6_K)?"Q6_K":ggml_type_name(ty);
        bool fits = (ne0%256==0 && ne1%16==0);
        int dmin_nz=-1, dmin_tot=0; float dmin_sample=0, d_sample=0;
        if((ty==GGML_TYPE_Q4_K||ty==GGML_TYPE_Q5_K) && fits){
            size_t toff = gguf_get_tensor_offset(g,i);
            int nb_row = (int)(ne0/256);
            int probe_rows = (ne1<8)?(int)ne1:8;        /* first 8 output rows */
            dmin_nz=0; dmin_tot=0;
            for(int r=0;r<probe_rows;r++){
                for(int b=0;b<nb_row;b++){
                    uint16_t dd,dm;
                    size_t bsz = (ty==GGML_TYPE_Q4_K)?sizeof(blk_q4_K):sizeof(blk_q5_K);
                    size_t abs = data_off + toff + ((size_t)r*nb_row+b)*bsz;
                    fseek(f, (long)abs, SEEK_SET);
                    uint8_t hdr[4]; fread(hdr,1,4,f);
                    memcpy(&dd,hdr,2); memcpy(&dm,hdr+2,2);
                    if(dm!=0){ if(dmin_nz==0){ dmin_sample=h2f(dm); d_sample=h2f(dd);} dmin_nz++; }
                    dmin_tot++;
                }
            }
        }
        if(ty==GGML_TYPE_Q4_K||ty==GGML_TYPE_Q5_K||ty==GGML_TYPE_Q6_K){
            printf("[%3lld] %-32s %-5s ne=[%lld,%lld,%lld,%lld] fits16x1=%s",
                   (long long)i, name, tn, (long long)ne0,(long long)ne1,(long long)ne2,(long long)ne3, fits?"Y":"n");
            if(dmin_nz>=0) printf("  dmin!=0: %d/%d  sample d=%.6g dmin=%.6g", dmin_nz, dmin_tot, d_sample, dmin_sample);
            printf("\n");
        }
    }
    fclose(f);
    gguf_free(g);
    return 0;
}
