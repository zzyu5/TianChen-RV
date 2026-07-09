/* iq4nl_gemm_paired_driver.c -- [l1-m2-iq4] iq4_nl repack GEMM prefill paired A/B on rvv/VLEN128.
 * OUR exported front-door-lowered repack GEMM (codebook vluxei16 gather) vs the OPPONENT's REAL
 * dispatched ggml_vec_dot_iq4_nl_q8_0 block-dot, linked from the board's own libggml-cpu.so.
 * Adapted from kquant_gemm_paired_q6k_driver.c. Control flow is data-independent (integer codebook
 * decode dominates, layouts/strides exact), so random fills give valid steady-state timing.
 * argv: <fmt=iq4_nl> <K(mult32)> <nr(mult4)> <nc(mult16)> <reps> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_NL 32
typedef uint16_t ggml_half;
typedef struct { ggml_half d; int8_t qs[QK_NL]; } block_q8_0; /* 34 */

/* OUR exported repack GEMM kernel (n, s, vx, vy, nr, nc, bs) */
extern void tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
/* OPPONENT: real ggml block-dot from libggml-cpu.so */
extern void ggml_vec_dot_iq4_nl_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p,size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64) { g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s iq4_nl K nr nc reps seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(strcmp(fmt,"iq4_nl")){ fprintf(stderr,"fmt must be iq4_nl\n"); return 2; }
    if(K%QK_NL||nr%4||nc%16){ fprintf(stderr,"K%%32, nr%%4, nc%%16 required\n"); return 2; }
    if(reps<10) reps=10;
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/QK_NL, grp_c=nc/16, grp_r=nr/4;

    size_t wstride = 288;   /* our block_iq4_nlx16 group stride (16 fp16 d + 16*16 nibbles) */
    size_t astride = 136;   /* our block_q8_0x4 group stride (4*34)                          */
    size_t wblk    = 18;    /* opponent plain block_iq4_nl (fp16 d + 16 qs)                   */
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;
    uint8_t* Wr=aligned_alloc(64,wbytes);
    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    uint8_t* Wo=aligned_alloc(64,(size_t)nc*nb*wblk);
    block_q8_0* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_0));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!Wr||!Ar||!Or||!Wo||!Ao||!Oo||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_rand(Wr,wbytes); fill_rand(Ar,abytes);
    fill_rand(Wo,(size_t)nc*nb*wblk); fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_0));
    memset(g_flush,1,FLUSH_BYTES);
    const uint16_t H=0x2C00; /* fp16 0.0625 */
    /* our repacked weight: 16 fp16 d at group start */
    for(int g=0;g<grp_c;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Wr+((size_t)g*nb+l)*wstride);
        for(int j=0;j<16;j++) p[j]=H; }
    /* our repacked activation: 4 fp16 d at group start */
    for(int g=0;g<grp_r;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Ar+((size_t)g*nb+l)*astride);
        for(int j=0;j<4;j++) p[j]=H; }
    /* opponent plain weight/activation fp16 d */
    for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* p=(uint16_t*)(Wo+i*wblk); p[0]=H; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=H;

    #define OUR_GEMM() tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc)
    #define OPP_GEMM() do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_0* ar=Ao+(size_t)r*nb; \
            ggml_vec_dot_iq4_nl_q8_0(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)

    OUR_GEMM(); OPP_GEMM(); /* warmup */

    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    volatile double sink=0;
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); OUR_GEMM(); ours[p]=now_ns()-t0; sink+=Or[0]+Or[(size_t)nr*nc-1];
        cold_flush(); double t1=now_ns(); OPP_GEMM(); opp[p] =now_ns()-t1; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }

    double macs=(double)nr*(double)nc*(double)K;
    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    printf("IQ4GEMM fmt=%s VLEN=%ld K=%d nr=%d nc=%d reps=%d noisefloor_ns=%.0f | "
           "OURS best_ns=%.0f med_ns=%.0f best_gmacs=%.4f | OPP best_ns=%.0f med_ns=%.0f best_gmacs=%.4f | "
           "ratio_best=%.3f ratio_med=%.3f sink=%.1f\n",
           fmt,vlen,K,nr,nc,reps,nf, ob,om,macs/ob, pb,pm,macs/pb, (macs/ob)/(macs/pb),(macs/om)/(macs/pm),(double)sink+(double)g_sink);
    free(Wr);free(Ar);free(Or);free(Wo);free(Ao);free(Oo);free(g_flush);free(ours);free(opp);
    return 0;
}
