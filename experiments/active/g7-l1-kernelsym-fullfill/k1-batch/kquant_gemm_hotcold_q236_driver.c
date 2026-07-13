/* kquant_gemm_hotcold_q236_driver.c — G7 §L1 k1-half: q2_K/q3_K/q6_K @k1/VLEN256 kernel-sym A/B.
 *
 * HOT+COLD same-shape GEMM micro. OURS = tcrv_emitc repack GEMM (PLAIN campaign export,
 * kq_export_q6q2q3: q2_K md5 6c9322f6 / q3_K f4cfb641 / q6_K 0f14791e) vs OPPONENT = board's own
 * factory block-dot ggml_vec_dot_q{2,3,6}_K_q8_K (machine-judged public T symbol), linked from
 * /data/k1build-stock libggml-cpu.so (clang++-18 -O3, symmetric with ours clang-18).
 *
 * Structure == experiments/active/g7-l2-kernelsym-hotcold/kquant_gemm_cold_driver.c (q4_K/q5_K),
 * generalized to q2/q3/q6 with the layout strides from kquant_gemm_paired_q6q2q3_driver.c:
 *   ours repacked block_qX_Kx16 group stride: q2=1344 q3=1824 q6=3360
 *   opponent plain block_qX_K:                q2=84   q3=110  q6=210
 *   ours activation block_q8_Kx4 group stride: 1168 ; opponent block_q8_K: 292
 * K-quant decode is DATA-INDEPENDENT (no data branches) so random fills give valid steady-state
 * GMAC/s. Correctness of ours is sealed separately at construction (bounded-norm oracle); this is a
 * THROUGHPUT probe (the two sides use different byte layouts, so no cross-check here — same
 * discipline as every kquant paired driver). [NG-4]: kernel-axis micro, NOT an e2e beat.
 *
 * HOT  = single reused weight tile, warmup + median-of-rounds of `iters` back-to-back calls.
 * COLD = POOL of P independent weight tiles (footprint >> k1 L2 512KiB), each round sweeps whole
 *        pool once so every weight tile is DRAM-cold on re-read. weight-GB/s reported for regime.
 *
 * argv: <fmt=q2_K|q3_K|q6_K> <K(mult256)> <nr(mult4)> <nc(mult16)> <pool_tiles> <rounds> <iters_hot> <seed>
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

extern void tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void tcrv_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
extern void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q3_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p,size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }
static void stats(double* v,int n,double*med,double*iqrpct){
  qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]);
  double q1=v[n/4],q3=v[(3*n)/4];
  *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0;
}

int main(int argc,char**argv){
    if(argc<9){ fprintf(stderr,"usage: %s q2_K|q3_K|q6_K K nr nc pool rounds iters seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),P=atoi(argv[5]),rounds=atoi(argv[6]),iters=atoi(argv[7]);
    rng=(uint64_t)strtoull(argv[8],0,0)|1ull;
    int f2=!strcmp(fmt,"q2_K"),f3=!strcmp(fmt,"q3_K"),f6=!strcmp(fmt,"q6_K");
    if(!f2&&!f3&&!f6){ fprintf(stderr,"fmt must be q2_K|q3_K|q6_K\n"); return 2; }
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8; if(iters<1)iters=1;
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wstride = f6?3360:(f2?1344:1824);   /* ours block_qX_Kx16 group stride */
    size_t astride = 1168;                     /* ours block_q8_Kx4 group stride  */
    size_t wblk    = f6?210:(f2?84:110);       /* opponent plain block_qX_K       */
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;
    size_t wobytes=(size_t)nc*nb*wblk;
    const uint16_t H=0x2C00;                   /* fp16 0.0625 */

    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    block_q8_K* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    uint8_t** Wr=malloc(P*sizeof(uint8_t*));
    uint8_t** Wo=malloc(P*sizeof(uint8_t*));
    if(!Ar||!Or||!Ao||!Oo||!Wr||!Wo){ fprintf(stderr,"OOM meta\n"); return 3; }
    fill_rand(Ar,abytes);
    fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_K));
    for(int g=0;g<grp_r;g++)for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride); for(int j=0;j<4;j++) p[j]=0.01f; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    for(int p=0;p<P;p++){
        Wr[p]=aligned_alloc(64,wbytes);
        Wo[p]=aligned_alloc(64,wobytes);
        if(!Wr[p]||!Wo[p]){ fprintf(stderr,"OOM pool %d\n",p); return 3; }
        fill_rand(Wr[p],wbytes); fill_rand(Wo[p],wobytes);
        for(int g=0;g<grp_c;g++)for(int l=0;l<nb;l++){ uint16_t* q=(uint16_t*)(Wr[p]+((size_t)g*nb+l)*wstride); for(int j=0;j<32;j++) q[j]=H; }
        for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* q=(uint16_t*)(Wo[p]+i*wblk); q[0]=H; if(f2) q[1]=H; }
    }
    size_t ws_ours=(size_t)P*wbytes + abytes + (size_t)nr*nc*sizeof(float);

    #define OURS_CALL(PI) do{ if(f2) tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Or,Wr[PI],Ar,(size_t)nr,(size_t)nc,(size_t)nc); \
        else if(f3) tcrv_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K((size_t)K,Or,Wr[PI],Ar,(size_t)nr,(size_t)nc,(size_t)nc); \
        else        tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K((size_t)K,Or,Wr[PI],Ar,(size_t)nr,(size_t)nc,(size_t)nc); }while(0)
    #define OPP_CALL(PI) do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo[PI]+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; \
            if(f2) ggml_vec_dot_q2_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); \
            else if(f3) ggml_vec_dot_q3_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); \
            else   ggml_vec_dot_q6_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)

    volatile double sink=0;
    double macs=(double)nr*(double)nc*(double)K;

    /* ---- HOT: single tile W[0], median-of-rounds of iters back-to-back ---- */
    for(int w=0;w<3;w++){ OURS_CALL(0); OPP_CALL(0); sink+=Or[0]+Oo[0]; }
    double *ho=malloc(rounds*sizeof(double)), *hp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int it=0;it<iters;it++){ OURS_CALL(0); sink+=Or[0]; } double t1=now_ns();
        ho[rd]=(t1-t0)/(double)iters;
        double t2=now_ns(); for(int it=0;it<iters;it++){ OPP_CALL(0); sink+=Oo[0]; } double t3=now_ns();
        hp[rd]=(t3-t2)/(double)iters;
    }
    double hom,hoi,hpm,hpi; stats(ho,rounds,&hom,&hoi); stats(hp,rounds,&hpm,&hpi);
    printf("HOT  fmt=%s VLEN=%ld K=%d nr=%d nc=%d iters=%d rounds=%d ours_ns=%.1f ours_iqr=%.2f ours_gmacs=%.4f "
           "opp_ns=%.1f opp_iqr=%.2f opp_gmacs=%.4f ratio_ours_over_opp=%.4f\n",
           fmt,vlen_bits,K,nr,nc,iters,rounds, hom,hoi,macs/hom, hpm,hpi,macs/hpm, (macs/hom)/(macs/hpm));

    /* ---- COLD: pool sweep, weights DRAM-cold ---- */
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){OURS_CALL(p);} sink+=Or[0]; }
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){OPP_CALL(p);}  sink+=Oo[0]; }
    double *co=malloc(rounds*sizeof(double)), *cp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++){OURS_CALL(p);} double t1=now_ns();
        co[rd]=(t1-t0)/(double)P; sink+=Or[0]+Or[(size_t)nr*nc-1];
        double t2=now_ns(); for(int p=0;p<P;p++){OPP_CALL(p);}  double t3=now_ns();
        cp[rd]=(t3-t2)/(double)P; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    double com,coi,cpm,cpi; stats(co,rounds,&com,&coi); stats(cp,rounds,&cpm,&cpi);
    double ours_wGBs=(double)wobytes/com, opp_wGBs=(double)wobytes/cpm;
    printf("COLD fmt=%s VLEN=%ld K=%d nr=%d nc=%d pool=%d ws_bytes=%zu wbytes_per_call=%zu "
           "ours_ns=%.1f ours_iqr=%.2f ours_gmacs=%.4f ours_wGBs=%.3f "
           "opp_ns=%.1f opp_iqr=%.2f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           fmt,vlen_bits,K,nr,nc,P,ws_ours,wobytes,
           com,coi,macs/com,ours_wGBs, cpm,cpi,macs/cpm,opp_wGBs, (macs/com)/(macs/cpm), (double)sink);
    return 0;
}
