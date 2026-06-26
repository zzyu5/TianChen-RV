// q4_1 repack GEMM (PREFILL) Win-A LMUL-width ablation: WIDE (i8m1->i16m2->i32m4->
// f32m4) vs NARROW (i8mf2->i16m1->i32m2->f32m2). BOTH compiler-emitted from the SAME
// tcrv_rvv.repack_gemm_q4_1_q8_1 op MLIR; the ONLY difference is the strip-width
// gearbox (default mf2 NARROW vs xtheadvector-stamp m1 WIDE) + the strip count it
// forces. Pure tune-ON(WIDE) vs tune-OFF(NARROW), both vectorized, same bytes read
// (N3-METHODOLOGY: NO scalar-contribution multiple). Gate = WIDE<->NARROW byte-exact.
// Repack/fill lifted verbatim from oracle_gemm_q41.cpp (GAP 2, already validated
// vs a scalar dequant-matmul oracle at norm <=7.6e-6).
//
// block_q4_1x16 (320): { fp16 d[16]; fp16 m[16]; uint8 qs[256]; }  weight qs@+64.
// block_q8_1x4  (144): { fp16 d[4];  fp16 s[4];  int8 qs[128]; }   act qs@+16, s@+8.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <riscv_vector.h>

#define QK 32
typedef uint16_t ggml_half;

struct block_q4_1 { ggml_half d; ggml_half m; uint8_t qs[QK/2]; };               // 20
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK]; };                  // 36
struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[QK*8]; };    // 320
struct block_q8_1x4  { ggml_half d[4];  ggml_half s[4];  int8_t qs[QK*4]; };     // 144

static inline ggml_half f2h(float fv){ _Float16 x=(_Float16)fv; ggml_half h; memcpy(&h,&x,2); return h; }
static inline float h2f(ggml_half h){ _Float16 x; memcpy(&x,&h,2); return (float)x; }

// the two compiler-EMITTED q4_1 repack GEMM kernels (renamed symbols)
#include "k_gemm_NARROW.cpp"
#include "k_gemm_WIDE.cpp"

static block_q4_1x16 make_w16(const block_q4_1* in){
    block_q4_1x16 o; for(int i=0;i<16;i++){o.d[i]=in[i].d;o.m[i]=in[i].m;}
    for(int i=0;i<QK*8;++i){int si=i%16,so=i/16; o.qs[i]=in[si].qs[so];}
    return o;
}
static block_q8_1x4 make_a4(const block_q8_1* in){
    block_q8_1x4 o; for(int c=0;c<4;c++){o.d[c]=in[c].d; o.s[c]=in[c].s;}
    for(int c=0;c<4;c++) for(int k=0;k<16;k++){ o.qs[k*4+c]=in[c].qs[k]; o.qs[64+k*4+c]=in[c].qs[16+k]; }
    return o;
}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double norm_cmp(const std::vector<float>&a,const std::vector<float>&b){
    double me=0,sse=0; size_t nn=a.size(); for(size_t j=0;j<nn;++j){ double v=b[j],e=fabs((double)a[j]-v); me=std::max(me,e); sse+=v*v; }
    double rms=sqrt(sse/nn); return rms>0?me/rms:me;
}

int main(int argc,char**argv){
    int n  = argc>1?atoi(argv[1]):4096;   // K (contraction)
    int nr = argc>2?atoi(argv[2]):16;     // activation rows (prefill tokens, mult of 4)
    int nc = argc>3?atoi(argv[3]):4096;   // output cols (weight rows, mult of 16)
    int reps=argc>4?atoi(argv[4]):200;
    int nb=n/QK; int ngrp=nc/16; int rgrp=nr/4;
    std::mt19937 rng(20260625);
    std::uniform_real_distribution<float> sc(0.005f,0.05f), mn(-0.2f,0.2f), act(-2.f,2.f);
    std::uniform_int_distribution<int> nib(0,15);

    // weights: nc rows x nb plain q4_1, repacked group-major.
    std::vector<block_q4_1> w((size_t)nc*nb);
    for(size_t i=0;i<w.size();++i){ w[i].d=f2h(sc(rng)); w[i].m=f2h(mn(rng)); for(int k=0;k<QK/2;++k) w[i].qs[k]=(uint8_t)(nib(rng)|(nib(rng)<<4)); }
    std::vector<block_q4_1x16> vx((size_t)ngrp*nb);
    for(int g=0;g<ngrp;++g){ block_q4_1 tmp[16]; for(int x=0;x<nb;++x){ for(int i=0;i<16;++i) tmp[i]=w[(size_t)(g*16+i)*nb+x]; vx[(size_t)g*nb+x]=make_w16(tmp); } }
    // activation: nr cols x n fp -> plain q8_1, repacked into q8_1x4 group-major.
    std::vector<float> af((size_t)nr*n); for(size_t i=0;i<af.size();++i) af[i]=act(rng);
    std::vector<block_q8_1> ay((size_t)nr*nb);
    for(int c=0;c<nr;++c) for(int b=0;b<nb;++b){ float amax=0; for(int j=0;j<QK;++j) amax=std::max(amax,fabsf(af[(size_t)c*n+b*QK+j])); float d=amax/127.f,id=d?1.f/d:0.f; int isum=0; for(int j=0;j<QK;++j){ int q=(int)lroundf(af[(size_t)c*n+b*QK+j]*id); ay[(size_t)c*nb+b].qs[j]=(int8_t)q; isum+=q;} ay[(size_t)c*nb+b].d=f2h(d); ay[(size_t)c*nb+b].s=f2h(d*(float)isum); }
    std::vector<block_q8_1x4> vy4((size_t)rgrp*nb);
    for(int g=0;g<rgrp;++g){ block_q8_1 t4[4]; for(int b=0;b<nb;++b){ for(int c=0;c<4;c++) t4[c]=ay[(size_t)(g*4+c)*nb+b]; vy4[(size_t)g*nb+b]=make_a4(t4);} }

    std::vector<float> oN((size_t)nr*nc), oW((size_t)nr*nc);
    k_gemm_q41_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc);
    k_gemm_q41_WIDE  ((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc);
    double nm=norm_cmp(oW,oN);
    printf("GEMM AGREE n=%d nr=%d nc=%d norm(WIDE vs NARROW)=%.3e %s\n",n,nr,nc,nm,nm<1e-4?"PASS":"FAIL");

    for(int i=0;i<10;++i){ k_gemm_q41_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc); k_gemm_q41_WIDE((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc);}
    double bN=1e18,bW=1e18;
    for(int r=0;r<reps;++r){
        double t0=now(); k_gemm_q41_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc); double tN=now()-t0; if(tN<bN)bN=tN;
        double t1=now(); k_gemm_q41_WIDE  ((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy4.data(),(size_t)nr,(size_t)nc,(size_t)nc); double tW=now()-t1; if(tW<bW)bW=tW;
    }
    printf("GEMM RESULT n=%d nr=%d nc=%d  NARROW=%.0f ns  WIDE=%.0f ns  ratio(NARROW/WIDE)=%.3fx\n",n,nr,nc,bN,bW,bN/bW);
    return 0;
}
