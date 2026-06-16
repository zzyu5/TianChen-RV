// INC-12 (q6_K K2) ggml-ABI link-compat test (ssh rvv, -ffp-contract=off).
//
// Proves the DEPLOYABLE drop-in: calls our kernel THROUGH the EXACT 8-arg ggml
// symbol `ggml_vec_dot_q6_K_q8_K(int n, float* s, size_t bs, const void* vx,
// size_t bx, const void* vy, size_t by, int nrc)` (via the tcrv_q6_k_q8_k.h
// wrapper) -- passing dummy bs/bx/by/nrc -- and compares *s byte-exact vs ggml's
// _generic. This exercises the real entry register-classes (which the 4-arg
// direct call in inc12_validate.cpp does NOT), so "drop-in" is actually tested.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#include "tcrv_q6_k_q8_k.h"  // the ggml-ABI bridge (8-arg ggml_vec_dot_q6_K_q8_K)

#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

static inline float fp16(uint16_t h){ _Float16 v; std::memcpy(&v,&h,2); return (float)v; }
#define GGML_CPU_FP16_TO_FP32(x) fp16(x)
#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)

// Verbatim ggml _generic (quants.c:800-853), same oracle as inc12_validate.cpp.
static void gen(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q6_K *x=(const block_q6_K*)vx; const block_q8_K *y=(const block_q8_K*)vy;
    const int nb=n/QK_K; int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    std::memset(sums,0,sizeof(sums)); float sumf=0;
    for(int i=0;i<nb;++i){ const uint8_t*q4=x[i].ql; const uint8_t*qh=x[i].qh; const int8_t*q8=y[i].qs;
        std::memset(aux32,0,sizeof(aux32)); int8_t*a=aux8;
        for(int j=0;j<QK_K;j+=128){ for(int l=0;l<32;++l){
            a[l+0]=(int8_t)((q4[l+0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
            a[l+32]=(int8_t)((q4[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
            a[l+64]=(int8_t)((q4[l+0]>>4)|(((qh[l]>>4)&3)<<4))-32;
            a[l+96]=(int8_t)((q4[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32; }
            a+=128; q4+=64; qh+=32; }
        a=aux8; int is=0;
        for(int j=0;j<QK_K/16;++j){ int scale=x[i].scales[is++];
            for(int l=0;l<8;++l)aux16[l]=q8[l]*a[l];
            for(int l=0;l<8;++l)aux32[l]+=scale*aux16[l]; q8+=8;a+=8;
            for(int l=0;l<8;++l)aux16[l]=q8[l]*a[l];
            for(int l=0;l<8;++l)aux32[l]+=scale*aux16[l]; q8+=8;a+=8; }
        const float d=GGML_CPU_FP16_TO_FP32(x[i].d)*y[i].d;
        for(int l=0;l<8;++l)sums[l]+=d*aux32[l]; }
    for(int l=0;l<8;++l)sumf+=sums[l]; *s=sumf;
}

static inline uint32_t bits(float f){ uint32_t u; std::memcpy(&u,&f,4); return u; }
static unsigned g=0x6b5f1c27u;
static unsigned r(){ g^=g<<13; g^=g>>17; g^=g<<5; return g; }
static uint16_t rd(){ uint16_t m=(uint16_t)(r()&0x3FF); uint16_t e=(uint16_t)(8+(r()%12)); return (uint16_t)((e<<10)|m); }
static void fb(uint8_t*q6k,uint8_t*q8k){
    for(int i=0;i<128;++i)q6k[i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<64;++i)q6k[128+i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<16;++i)q6k[192+i]=(uint8_t)(r()&0xFF);
    uint16_t d=rd(); std::memcpy(q6k+208,&d,2);
    float dy=(float)((int)(r()%2000)-1000)/4096.0f; std::memcpy(q8k+0,&dy,4);
    for(int i=0;i<256;++i)q8k[4+i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<32;++i)q8k[260+i]=0;
}

int main(){
    int failures=0, checked=0;
    printf("INC-12 q6_K K2 ggml-ABI link-compat: 8-arg ggml_vec_dot_q6_K_q8_K -> our kernel\n");
    const int ns[]={1,2,8,16,100};
    const char*nl[]={"n=256","n=512","n=2048","n=4096","n=25600"};
    for(int i=0;i<5;++i){ int NB=ns[i]; size_t n=(size_t)NB*256;
        uint8_t*vx=(uint8_t*)malloc((size_t)NB*210); uint8_t*vy=(uint8_t*)malloc((size_t)NB*292);
        for(int ib=0;ib<NB;++ib) fb(vx+(size_t)ib*210, vy+(size_t)ib*292);
        float ref=0, got=0;
        gen((int)n,&ref,0,vx,0,vy,0,1);
        // Call THROUGH the 8-arg ggml entry with dummy bs/bx/by/nrc.
        ggml_vec_dot_q6_K_q8_K((int)n,&got,/*bs*/123,vx,/*bx*/45,vy,/*by*/67,/*nrc*/1);
        int f=(bits(ref)!=bits(got)); failures+=f; ++checked;
        printf("  [%s] ref=%.9g (0x%08x) tcrv(8-arg)=%.9g (0x%08x) %s\n",
               nl[i],ref,bits(ref),got,bits(got), f?"MISMATCH":"bit-exact");
        free(vx); free(vy);
    }
    // Many random n through the 8-arg entry.
    int rndfail=0;
    for(int t=0;t<1000;++t){ int NB=1+(int)(r()%32); size_t n=(size_t)NB*256;
        uint8_t*vx=(uint8_t*)malloc((size_t)NB*210); uint8_t*vy=(uint8_t*)malloc((size_t)NB*292);
        for(int ib=0;ib<NB;++ib) fb(vx+(size_t)ib*210, vy+(size_t)ib*292);
        float ref=0, got=0;
        gen((int)n,&ref,0,vx,0,vy,0,1);
        ggml_vec_dot_q6_K_q8_K((int)n,&got,99,vx,88,vy,77,1);
        if(bits(ref)!=bits(got)) ++rndfail; ++checked;
        free(vx); free(vy);
    }
    failures+=rndfail;
    printf("  random-through-8-arg-entry mismatches: %d / 1000\n", rndfail);
    printf("\nINC-12 link-compat: %d cases, %d failures\n", checked, failures);
    printf("RESULT: %s\n", failures==0 ? "PASS (drop-in 8-arg ggml entry bit-exact vs _generic)" : "FAIL");
    return failures==0?0:1;
}
