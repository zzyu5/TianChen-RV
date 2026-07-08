/* kquant_gemm_paired_q6k_driver.c -- [KQUANT-L1 / T3-tile] q6_K prefill paired A/B on rvv/VLEN128.
 * q6_K-ONLY trim of kquant_gemm_paired_q6q2q3_driver.c: OUR exported repack GEMM (front-door
 * lowered) vs the OPPONENT's REAL dispatched path ggml_vec_dot_q6_K_q8_K, linked directly from
 * the board's own libggml-cpu.so. At VLEN128 ggml's repack trait selector returns NULLPTR for
 * q6_K (NEON-only), so ggml's prefill mul_mat falls back to exactly this per-(row,col) block-dot.
 *
 * OUR q6_K symbol is the SAME for the un-tiled (HEAD 924dc31f export) and the tiled (working-tree
 * S6 stack-panel) kernels, so this driver is linked TWICE (untiled.o / tiled.o) into two binaries
 * that differ ONLY in the q6_K GEMM object -> a clean tiling-isolated A/B.
 *
 * Protocol: PAIRED-external (this driver times ONE side; the harness runs untiled then tiled as
 * fresh procs per cold round). N>=10 reps, COLD-CACHE flush (224 MiB, >3x L3=64MiB) before EACH
 * timed region, single full-GEMM call per timed region, report best+median GMAC/s + ratio + T-N
 * noise floor. Control flow is data-independent, so random fills give valid steady-state timing.
 *
 * IDENTITY-DUMP mode: if env Q6K_DUMP=<path> is set, after warmup run OUR_GEMM once and fwrite the
 * full nr*nc fp32 output to <path>, then exit -- lets the harness cmp untiled-vs-tiled byte-exactly
 * on the real small-finite fp16-scale fold path (the two binaries share this identical fill).
 *
 * [NG-4] DISCIPLINE: L1 path-candidate datapoint (opponent has no working repack@128, we do).
 * NOT a [PERF-1] eight-gate beat -- single-core, opponent = single-thread block-dot loop
 * (kernel-axis proxy, not threaded mul_mat). Reported as a gap, not a win.
 *
 * argv: <fmt=q6_K> <K(mult256)> <nr(mult4)> <nc(mult16)> <reps> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K; /* 292 */

/* OUR exported repack GEMM kernel (n, s, vx, vy, nr, nc, bs) -- symbol shared untiled/tiled */
extern void tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
/* OPPONENT: real ggml block-dot, linked from libggml-cpu.so */
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p,size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

/* cold-cache flush: stream a >3x L3 buffer (224 MiB) */
#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64) { g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s q6_K K nr nc reps seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(strcmp(fmt,"q6_K")){ fprintf(stderr,"fmt must be q6_K\n"); return 2; }
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(reps<10) reps=10;
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wstride = 3360;                             /* our block_q6_Kx16 group stride */
    size_t astride = 1168;                             /* our block_q8_Kx4 group stride  */
    size_t wblk    = 210;                              /* opponent plain block_q6_K      */
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;
    uint8_t* Wr=aligned_alloc(64,wbytes);
    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    uint8_t* Wo=aligned_alloc(64,(size_t)nc*nb*wblk);
    block_q8_K* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!Wr||!Ar||!Or||!Wo||!Ao||!Oo||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_rand(Wr,wbytes); fill_rand(Ar,abytes);
    fill_rand(Wo,(size_t)nc*nb*wblk); fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_K));
    memset(g_flush,1,FLUSH_BYTES);
    /* keep fp scale fields finite-small so the DCE sink stays a real number (integer decode
     * dominates runtime and is data-independent; layouts/strides are exact). */
    const uint16_t H=0x2C00; /* fp16 0.0625 */
    for(int g=0;g<grp_c;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Wr+((size_t)g*nb+l)*wstride);
        for(int j=0;j<32;j++) p[j]=H; }  /* d strip small-finite */
    for(int g=0;g<grp_r;g++)for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride);
        for(int j=0;j<4;j++) p[j]=0.01f; }
    for(size_t i=0;i<(size_t)nc*nb;i++){ /* opponent block scale field (fp16 d at start) */
        uint16_t* p=(uint16_t*)(Wo+i*wblk); p[0]=H; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    #define OUR_GEMM() tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc)
    #define OPP_GEMM() do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; \
            ggml_vec_dot_q6_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)

    /* warmup (not timed) */
    OUR_GEMM(); OPP_GEMM();

    /* IDENTITY-DUMP mode: emit full fp32 output for a byte-exact untiled-vs-tiled cmp, then exit. */
    const char* dump=getenv("Q6K_DUMP");
    if(dump && *dump){
        OUR_GEMM();
        FILE* f=fopen(dump,"wb");
        if(!f){ fprintf(stderr,"dump open fail %s\n",dump); return 4; }
        size_t nel=(size_t)nr*nc;
        if(fwrite(Or,sizeof(float),nel,f)!=nel){ fprintf(stderr,"dump write fail\n"); fclose(f); return 5; }
        fclose(f);
        printf("Q6K_DUMP wrote %zu floats to %s (nr=%d nc=%d)\n",nel,dump,nr,nc);
        return 0;
    }

    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    volatile double sink=0;
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); OUR_GEMM(); ours[p]=now_ns()-t0; sink+=Or[0]+Or[(size_t)nr*nc-1];
        cold_flush(); double t1=now_ns(); OPP_GEMM(); opp[p] =now_ns()-t1; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    /* T-N noise floor: empty timed region around a cold flush */
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }

    double macs=(double)nr*(double)nc*(double)K;
    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    printf("KQGEMM fmt=%s VLEN=%ld K=%d nr=%d nc=%d reps=%d noisefloor_ns=%.0f | "
           "OURS best_ns=%.0f med_ns=%.0f best_gmacs=%.4f | OPP best_ns=%.0f med_ns=%.0f best_gmacs=%.4f | "
           "ratio_best=%.3f ratio_med=%.3f sink=%.1f\n",
           fmt,vlen,K,nr,nc,reps,nf, ob,om,macs/ob, pb,pm,macs/pb, (macs/ob)/(macs/pb),(macs/om)/(macs/pm),(double)sink+(double)g_sink);
    free(Wr);free(Ar);free(Or);free(Wo);free(Ao);free(Oo);free(g_flush);free(ours);free(opp);
    return 0;
}
