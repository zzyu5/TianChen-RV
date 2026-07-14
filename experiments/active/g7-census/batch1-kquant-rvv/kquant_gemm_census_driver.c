/* kquant_gemm_census_driver.c — G7 L1 Batch-1 K-quant@rvv cold-start census (5 formats).
 *
 * ONE binary, ONE process, SAME session/core: measures HOT (single-buffer, warmup+best-of-N)
 * AND COLD (224 MiB flush before EACH timed region, paired A/B, N reps, best+median+relIQR)
 * for OUR exported repack-GEMM kernel vs the OPPONENT's REAL dispatched path.
 *
 * Merges the two VALIDATED drivers (kquant_gemm_paired_driver.c hot best-of + q6q2q3
 * cold-flush paired) into a single 5-format sweep; strides/init IDENTICAL to those drivers
 * (byte-verified against the fixture weight_block_stride/activation_block_stride attrs:
 * q2=1344 q3=1824 q4=2304 q5=2816 q6=3360, all act=1168).
 *
 * OPPONENT (machine-judged @VLEN128, commit f3e1828): repack GEMM trait selector returns
 * NULLPTR for ALL 5 K-quants (q4/q5/q2 case-128=TODO, q6/q3 no riscv branch) => prefill
 * mul_mat falls back to the per-(row,col) block-dot ggml_vec_dot_qX_K_q8_K, linked from the
 * board's own libggml-cpu.so. That IS the fair dispatched baseline (the ggml_gemm_qX_K_16x1/
 * 8x8 kernels EXIST but are VLEN256-gated-off / NEON-only => disclosed stronger unused path).
 *
 * Control flow of both sides is DATA-INDEPENDENT (K-quant decode has no data branches), so
 * random byte fills give valid steady-state timing. Correctness is proven SEPARATELY on
 * silicon (per-format repack verifiers, INT byte-exact + bounded-norm). [NG-4] kernel-axis
 * datapoint, NOT an e2e beat, NOT a sealed 8-gate Win. Main tree + build/ UNTOUCHED.
 *
 * argv: <fmt=q2_K|q3_K|q4_K|q5_K|q6_K> <K(mult256)> <nr(mult4)> <nc(mult16)> <hot_iters> <cold_reps> <seed>
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

/* OUR exported repack GEMM kernels (weft_emitc; n, s, vx=weights, vy=activations, nr, nc, bs) */
extern void weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void weft_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
/* OPPONENT: real ggml block-dot, linked from libggml-cpu.so */
extern void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q3_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p,size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

/* cold-cache flush: stream a >3x L3 (64MiB) buffer => 224 MiB */
#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

int main(int argc,char**argv){
    if(argc<8){ fprintf(stderr,"usage: %s q2_K|q3_K|q4_K|q5_K|q6_K K nr nc hot_iters cold_reps seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),hiters=atoi(argv[5]),reps=atoi(argv[6]);
    rng=(uint64_t)strtoull(argv[7],0,0)|1ull;
    int fi = !strcmp(fmt,"q2_K")?2 : !strcmp(fmt,"q3_K")?3 : !strcmp(fmt,"q4_K")?4 : !strcmp(fmt,"q5_K")?5 : !strcmp(fmt,"q6_K")?6 : 0;
    if(!fi){ fprintf(stderr,"fmt must be q{2,3,4,5,6}_K\n"); return 2; }
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wstride = fi==2?1344 : fi==3?1824 : fi==4?2304 : fi==5?2816 : 3360; /* block_qX_Kx16 group */
    size_t astride = 1168;                                                     /* block_q8_Kx4 group   */
    size_t wblk    = fi==2?84   : fi==3?110  : fi==4?144  : fi==5?176  : 210;  /* opponent plain block */
    int has_dmin   = (fi==2||fi==4||fi==5);                                     /* q2/q4/q5 carry dmin  */
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
    /* finite-small fp scale fields so the DCE sink stays a real number (integer decode+vwmacc
     * dominates runtime and is data-independent; layouts/strides are exact). */
    const uint16_t H=0x2C00; /* fp16 0.0625 */
    for(int g=0;g<grp_c;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Wr+((size_t)g*nb+l)*wstride); for(int j=0;j<32;j++) p[j]=H; }
    for(int g=0;g<grp_r;g++)for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride); for(int j=0;j<4;j++) p[j]=0.01f; }
    for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* p=(uint16_t*)(Wo+i*wblk); p[0]=H; if(has_dmin) p[1]=H; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    #define OUR_GEMM() do{ switch(fi){ \
        case 2: weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc); break; \
        case 3: weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc); break; \
        case 4: weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc); break; \
        case 5: weft_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc); break; \
        default:weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc); } }while(0)
    #define OPP_GEMM() do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; float* o=&Oo[(size_t)r*nc+c]; \
            switch(fi){ case 2: ggml_vec_dot_q2_K_q8_K(K,o,0,wc,0,ar,0,1); break; \
                        case 3: ggml_vec_dot_q3_K_q8_K(K,o,0,wc,0,ar,0,1); break; \
                        case 4: ggml_vec_dot_q4_K_q8_K(K,o,0,wc,0,ar,0,1); break; \
                        case 5: ggml_vec_dot_q5_K_q8_K(K,o,0,wc,0,ar,0,1); break; \
                        default:ggml_vec_dot_q6_K_q8_K(K,o,0,wc,0,ar,0,1); } } } }while(0)

    double macs=(double)nr*(double)nc*(double)K;
    volatile double sink=0;

    /* ================= HOT (single buffer, warmup + best-of-N passes of hiters calls) ============= */
    for(int w=0;w<3;w++){ OUR_GEMM(); } sink+=Or[0];
    double ours_hot=1e30;
    for(int pass=0;pass<8;pass++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ OUR_GEMM(); } double npc=(now_ns()-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; sink+=Or[0]+Or[(size_t)nr*nc-1]; }
    for(int w=0;w<2;w++){ OPP_GEMM(); } sink+=Oo[0];
    double opp_hot=1e30;
    for(int pass=0;pass<8;pass++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ OPP_GEMM(); } double npc=(now_ns()-t0)/(double)hiters; if(npc<opp_hot)opp_hot=npc; sink+=Oo[0]+Oo[(size_t)nr*nc-1]; }

    /* ================= COLD (224 MiB flush before EACH timed region; paired A/B; N reps) ========== */
    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    /* warmup (bring code/TLB steady; data re-flushed cold each rep) */
    cold_flush(); OUR_GEMM(); cold_flush(); OPP_GEMM();
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); OUR_GEMM(); ours[p]=now_ns()-t0; sink+=Or[0]+Or[(size_t)nr*nc-1];
        cold_flush(); double t1=now_ns(); OPP_GEMM(); opp[p] =now_ns()-t1; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    /* T-N noise floor: empty timed region around a cold flush */
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }

    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    double o_q1=ours[reps/4], o_q3=ours[(3*reps)/4], p_q1=opp[reps/4], p_q3=opp[(3*reps)/4];
    double o_iqr = om>0?100.0*(o_q3-o_q1)/om:0.0, p_iqr = pm>0?100.0*(p_q3-p_q1)/pm:0.0;

    /* weight footprint GB/s (plain block footprint, comparable both sides; regime classify) */
    double wob=(double)nc*nb*wblk;
    double ours_cold_gbs=wob/om, opp_cold_gbs=wob/pm;

    printf("CENSUS fmt=%s VLEN=%ld K=%d nr=%d nc=%d hiters=%d reps=%d nf_ns=%.0f | "
           "HOT ours_ns=%.0f opp_ns=%.0f ours_gmacs=%.4f opp_gmacs=%.4f ratio_hot=%.4f | "
           "COLD ours_med_ns=%.0f ours_iqrpct=%.2f opp_med_ns=%.0f opp_iqrpct=%.2f "
           "ours_gmacs=%.4f opp_gmacs=%.4f ours_wGBs=%.3f opp_wGBs=%.3f ratio_cold_med=%.4f ratio_cold_best=%.4f | "
           "sink=%.1f\n",
           fmt,vlen,K,nr,nc,hiters,reps,nf,
           ours_hot,opp_hot,macs/ours_hot,macs/opp_hot,(macs/ours_hot)/(macs/opp_hot),
           om,o_iqr,pm,p_iqr,macs/om,macs/pm,ours_cold_gbs,opp_cold_gbs,(macs/om)/(macs/pm),(macs/ob)/(macs/pb),
           (double)sink+(double)g_sink);
    free(Wr);free(Ar);free(Or);free(Wo);free(Ao);free(Oo);free(g_flush);free(ours);free(opp);
    return 0;
}
