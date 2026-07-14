/* q6k_roll_ab_driver.c — G7 L1' q6_K@rvv GEMM whole-K-nest roll: on-silicon
 * byte-exact A/B (ROLLED vs UNROLLED, mismatch=0) + 3-way cold GEMM prefill timing
 * (ROLLED vs UNROLLED(current shipped) vs STOCK ggml block-dot).
 *
 * Adapted from the VALIDATED kquant_gemm_census_driver.c (SAME strides/init/cold-flush
 * paired machinery), specialized to q6_K and extended to link BOTH our q6_K GEMM
 * emitters (unrolled + rolled, distinct symbols) so the roll's numeric equivalence is
 * proven ON HARDWARE (not merely by-construction) and the recovery ratio measured.
 *
 * OPPONENT (machine-judged @VLEN128, same as batch1 census): repack GEMM trait selector
 * returns NULLPTR for q6_K => prefill mul_mat falls back to per-(row,col) block-dot
 * ggml_vec_dot_q6_K_q8_K, linked from the board's own libggml-cpu.so. Fair dispatched
 * baseline. Control flow data-independent => random fills give valid steady-state timing.
 *
 * argv: <K(mult256)> <nr(mult4)> <nc(mult16)> <hot_iters> <cold_reps> <seed>
 * [NG-4] kernel-axis datapoint; sub-parity recovery, NOT an e2e beat / sealed Win.
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
_Static_assert(sizeof(block_q8_K)==292,"q8_K");

/* OUR two q6_K repack GEMM emitters (n, s, vx=weights, vy=activations, nr, nc, bs) */
extern void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K_ROLLED(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
/* OPPONENT: real ggml block-dot, from libggml-cpu.so */
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p,size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s K nr nc hot_iters cold_reps seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]),nr=atoi(argv[2]),nc=atoi(argv[3]),hiters=atoi(argv[4]),reps=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wstride=3360, astride=1168, wblk=210; /* q6_K: block_q6_Kx16 / block_q8_Kx4 / opponent plain */
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;

    uint8_t* Wr=aligned_alloc(64,wbytes);
    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Ou=aligned_alloc(64,(size_t)nr*nc*sizeof(float)); /* unrolled out */
    float*   Ol=aligned_alloc(64,(size_t)nr*nc*sizeof(float)); /* rolled out   */
    uint8_t* Wo=aligned_alloc(64,(size_t)nc*nb*wblk);
    block_q8_K* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float)); /* opponent out */
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!Wr||!Ar||!Ou||!Ol||!Wo||!Ao||!Oo||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_rand(Wr,wbytes); fill_rand(Ar,abytes);
    fill_rand(Wo,(size_t)nc*nb*wblk); fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_K));
    memset(g_flush,1,FLUSH_BYTES);
    const uint16_t H=0x2C00; /* fp16 0.0625 */
    for(int g=0;g<grp_c;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Wr+((size_t)g*nb+l)*wstride); for(int j=0;j<32;j++) p[j]=H; }
    for(int g=0;g<grp_r;g++)for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride); for(int j=0;j<4;j++) p[j]=0.01f; }
    for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* p=(uint16_t*)(Wo+i*wblk); p[0]=H; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    #define UNROLLED() weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K((size_t)K,Ou,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc)
    #define ROLLED()   weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K_ROLLED((size_t)K,Ol,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc)
    #define OPP() do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; float* o=&Oo[(size_t)r*nc+c]; \
            ggml_vec_dot_q6_K_q8_K(K,o,0,wc,0,ar,0,1); } } }while(0)

    /* ===== BYTE-EXACT A/B on silicon: ROLLED vs UNROLLED on IDENTICAL repacked input ===== */
    memset(Ou,0,(size_t)nr*nc*sizeof(float)); memset(Ol,0,(size_t)nr*nc*sizeof(float));
    UNROLLED(); ROLLED();
    int mm=memcmp(Ou,Ol,(size_t)nr*nc*sizeof(float));
    int ndiff=0; float maxabs=0.f;
    for(size_t i=0;i<(size_t)nr*nc;i++){ if(Ou[i]!=Ol[i]){ ndiff++; float d=Ou[i]-Ol[i]; if(d<0)d=-d; if(d>maxabs)maxabs=d; } }
    int nz=0; for(size_t i=0;i<(size_t)nr*nc;i++) if(Ou[i]!=0.f) nz++;
    printf("BYTEEXACT q6_K roll-vs-unroll: memcmp=%d (0=byte-exact) ndiff=%d/%d maxabsdiff=%.6g nonzero=%d\n",
           mm,ndiff,nr*nc,(double)maxabs,nz);

    double macs=(double)nr*(double)nc*(double)K;
    volatile double sink=0;

    /* ===== HOT best-of-N (warmup + best) for all three ===== */
    for(int w=0;w<3;w++){ UNROLLED(); } sink+=Ou[0];
    double u_hot=1e30; for(int pass=0;pass<8;pass++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ UNROLLED(); } double n=(now_ns()-t0)/hiters; if(n<u_hot)u_hot=n; sink+=Ou[0]; }
    for(int w=0;w<3;w++){ ROLLED(); } sink+=Ol[0];
    double r_hot=1e30; for(int pass=0;pass<8;pass++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ ROLLED(); } double n=(now_ns()-t0)/hiters; if(n<r_hot)r_hot=n; sink+=Ol[0]; }
    for(int w=0;w<2;w++){ OPP(); } sink+=Oo[0];
    double p_hot=1e30; for(int pass=0;pass<8;pass++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ OPP(); } double n=(now_ns()-t0)/hiters; if(n<p_hot)p_hot=n; sink+=Oo[0]; }

    /* ===== COLD paired (224MiB flush before EACH timed region; N reps; median) ===== */
    double* U=malloc(sizeof(double)*reps);
    double* R=malloc(sizeof(double)*reps);
    double* P=malloc(sizeof(double)*reps);
    cold_flush(); UNROLLED(); cold_flush(); ROLLED(); cold_flush(); OPP();
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); UNROLLED(); U[p]=now_ns()-t0; sink+=Ou[0]+Ou[(size_t)nr*nc-1];
        cold_flush(); double t1=now_ns(); ROLLED();   R[p]=now_ns()-t1; sink+=Ol[0]+Ol[(size_t)nr*nc-1];
        cold_flush(); double t2=now_ns(); OPP();       P[p]=now_ns()-t2; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }
    double ub=1e30,rb=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(U[p]<ub)ub=U[p]; if(R[p]<rb)rb=R[p]; if(P[p]<pb)pb=P[p]; }
    qsort(U,reps,sizeof(double),cmp_d); qsort(R,reps,sizeof(double),cmp_d); qsort(P,reps,sizeof(double),cmp_d);
    double um=U[reps/2], rm=R[reps/2], pm=P[reps/2];
    double u_iqr=um>0?100.0*(U[(3*reps)/4]-U[reps/4])/um:0;
    double r_iqr=rm>0?100.0*(R[(3*reps)/4]-R[reps/4])/rm:0;
    double p_iqr=pm>0?100.0*(P[(3*reps)/4]-P[reps/4])/pm:0;

    printf("Q6KROLL VLEN=%ld K=%d nr=%d nc=%d hiters=%d reps=%d nf_ns=%.0f | "
           "HOT unroll_ns=%.0f roll_ns=%.0f opp_ns=%.0f roll/unroll=%.4f roll/opp=%.4f unroll/opp=%.4f | "
           "COLD unroll_med=%.0f(iqr%.1f%%) roll_med=%.0f(iqr%.1f%%) opp_med=%.0f(iqr%.1f%%) "
           "recovery_roll/unroll_med=%.4f roll/opp_med=%.4f unroll/opp_med=%.4f "
           "recovery_best=%.4f roll/opp_best=%.4f unroll/opp_best=%.4f | sink=%.1f\n",
           vlen,K,nr,nc,hiters,reps,nf,
           u_hot,r_hot,p_hot,u_hot/r_hot,p_hot/r_hot,p_hot/u_hot,
           um,u_iqr,rm,r_iqr,pm,p_iqr,
           um/rm, pm/rm, pm/um,
           ub/rb, pb/rb, pb/ub,
           (double)sink+(double)g_sink);
    free(Wr);free(Ar);free(Ou);free(Ol);free(Wo);free(Ao);free(Oo);free(g_flush);free(U);free(R);free(P);
    return 0;
}
