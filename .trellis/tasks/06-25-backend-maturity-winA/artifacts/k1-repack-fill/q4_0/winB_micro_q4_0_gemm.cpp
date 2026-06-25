// Win-B micro (prefill GEMM): OUR compiler-emitted q4_0 repack GEMM (16x1, nr=4
// rows x nc=16 cols, block-as-lane) vs ggml's REAL shipped vectorized
// ggml_gemm_q4_0_16x1_q8_0 (same x16 repack layout, ggml's own VLEN-vectorized
// GEMM) -- the methodology-correct Win-B baseline (ggml SHIPS this repack GEMM,
// so the baseline is its real vectorized kernel, NOT scalar/_generic).
// Regime: k1 (Spacemit X60), VLEN256, clang-18 -O3, taskset -c 0, best-of-reps min.
// SANITY ours==ggml byte-exact BEFORE any ratio.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>

#define QK4_0 32
#define QK8_0 32
typedef uint16_t ggml_half;
static inline ggml_half f2h(float fv){uint32_t x;memcpy(&x,&fv,4);uint32_t s=(x>>16)&0x8000;int32_t e=((x>>23)&0xff)-127+15;uint32_t m=x&0x7fffff;if(e<=0)return(ggml_half)s;if(e>=0x1f)return(ggml_half)(s|0x7c00);uint32_t h=s|(e<<10)|(m>>13);if((m&0x1000)&&((m&0x2fff)||(h&1)))h++;return(ggml_half)h;}
struct block_q4_0   { ggml_half d; uint8_t qs[QK4_0/2]; };
struct block_q4_0x16{ ggml_half d[16]; int8_t qs[QK8_0*8]; };  // 288
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[QK8_0*4]; };   // 136

extern "C" {
void OURS_gemm_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
void ggml_gemm_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
void ggml_quantize_mat_q8_0_4x1(const float*, void*, int64_t);
}

static block_q4_0x16 make_block(const block_q4_0* in){
    block_q4_0x16 out;
    for(int i=0;i<16;i++) out.d[i]=in[i].d;
    const int end=QK4_0*8; const uint8_t xm=0x88;
    for(int i=0;i<end;++i){int sid=i%16,so=i/16; out.qs[i]=(int8_t)(in[sid].qs[so]^xm);}
    return out;
}
static double now(){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}

int main(){
    const int n=64*QK4_0, nb=n/QK4_0, NC=16;       // n=2048
    std::mt19937 rng(20260625);
    std::uniform_int_distribution<int> nib(0,15);
    std::uniform_real_distribution<float> sc(0.005f,0.05f), act(-2.f,2.f);
    // weights: 16 rows x nb plain blocks -> one x16 stream
    block_q4_0* w=(block_q4_0*)malloc((size_t)NC*nb*sizeof(block_q4_0));
    for(int r=0;r<NC;r++)for(int b=0;b<nb;b++){w[r*nb+b].d=f2h(sc(rng));for(int k=0;k<QK4_0/2;k++)w[r*nb+b].qs[k]=(uint8_t)(nib(rng)|(nib(rng)<<4));}
    block_q4_0x16* vx=(block_q4_0x16*)malloc((size_t)nb*sizeof(block_q4_0x16));
    {block_q4_0 tmp[16];for(int b=0;b<nb;b++){for(int i=0;i<16;i++)tmp[i]=w[i*nb+b];vx[b]=make_block(tmp);}}
    // 4 activation rows -> q8_0x4
    float* af=(float*)malloc((size_t)4*n*sizeof(float));
    for(int i=0;i<4*n;i++)af[i]=act(rng);
    block_q8_0x4* vy=(block_q8_0x4*)malloc((size_t)nb*sizeof(block_q8_0x4));
    ggml_quantize_mat_q8_0_4x1(af,vy,n);
    const size_t bs=16;
    float o_ours[4*16], o_ggml[4*16];
    OURS_gemm_q4_0_16x1_q8_0(n,o_ours,bs,vx,vy,4,NC);
    ggml_gemm_q4_0_16x1_q8_0(n,o_ggml,bs,vx,vy,4,NC);
    double maxrel=0,sse=0;long c=0;
    for(int m=0;m<4;m++)for(int j=0;j<16;j++){double v=o_ggml[m*bs+j];double e=fabs((double)o_ours[m*bs+j]-v);double dn=fabs(v);if(dn<1e-6)dn=1e-6;double rl=e/dn;if(rl>maxrel)maxrel=rl;sse+=v*v;c++;}
    printf("SANITY max_rel=%.3e rms=%.3e (ours-GEMM h16 vs ggml real 16x1 GEMM)\n",maxrel,sqrt(sse/c));
    const int iters=4000,reps=200;
    for(int i=0;i<iters;i++){float s[64];OURS_gemm_q4_0_16x1_q8_0(n,s,bs,vx,vy,4,NC);ggml_gemm_q4_0_16x1_q8_0(n,s,bs,vx,vy,4,NC);}
    double ob=1e18,gb=1e18;
    for(int r=0;r<reps;r++){
        double t0=now();for(int i=0;i<iters;i++){float s[64];OURS_gemm_q4_0_16x1_q8_0(n,s,bs,vx,vy,4,NC);}double po=(now()-t0)/iters;if(po<ob)ob=po;
        double t1=now();for(int i=0;i<iters;i++){float s[64];ggml_gemm_q4_0_16x1_q8_0(n,s,bs,vx,vy,4,NC);}double pg=(now()-t1)/iters;if(pg<gb)gb=pg;
    }
    printf("RESULT ours(repack-GEMM h16,4x16) %.1f ns\n",ob);
    printf("RESULT ggml(real 16x1 GEMM,4x16)  %.1f ns\n",gb);
    printf("RATIO ggml/ours %.3f  (>1 = our repack WIN)\n",gb/ob);
    free(w);free(vx);free(af);free(vy);
    return 0;
}
