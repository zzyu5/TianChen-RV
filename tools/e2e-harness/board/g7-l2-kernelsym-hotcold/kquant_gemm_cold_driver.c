/* kquant_gemm_cold_driver.c — G7 L2 货架A cold micro A/B (q4_K / q5_K @k1/VLEN256).
 *
 * COLD variant of kquant_gemm_paired_driver.c. Difference from the HOT driver:
 *   HOT  = one (W,A,O) buffer set, warmup + best-of-5 of iters back-to-back calls
 *          over the SAME buffers  =>  weight matrix re-read from L2 (cache-WARM-ish;
 *          note k1 L2=512KiB < one q4_K weight tile 590KiB, so hot is only partly warm).
 *   COLD = a POOL of P independent WEIGHT buffers whose combined footprint >> LLC
 *          (k1 L2 = 512 KiB per cluster). Each timed ROUND sweeps the ENTIRE pool once
 *          (one call per tile); by the time tile 0 is revisited next round it has been
 *          evicted, so every weight tile is read COLD from DRAM. Activation+output are
 *          single/reused (small; in decode the ACTIVATION is the warm current token while
 *          WEIGHTS stream cold — this driver models exactly that). median-of-N rounds.
 *
 * This isolates the CACHE-RESIDENCY variable at a fixed kernel+shape (same kernel, same
 * opponent, same K/nr/nc as the hot measurement). Reports achieved weight-GB/s so the
 * regime (compute-bound prefill vs memory-bound decode) can be roofline-classified.
 *
 * Opponent = board's own libggml-cpu.so ggml_vec_dot_q{4,5}_K_q8_K block-dot (SAME opponent
 * as the hot t4a measurement, so hot/cold ratios are apples-to-apples). [NG-4]: kernel-axis
 * datapoint, NOT an e2e beat, NOT a sealed 8-gate Win. Correctness is sealed separately
 * (golden==S6 byte-identity @VLEN256, t4a); this is a THROUGHPUT probe on data-independent
 * control flow (K-quant decode has no data branches), independent random fills per tile.
 *
 * argv: <fmt=q4_K|q5_K> <K(mult256)> <nr(mult4)> <nc(mult16)> <pool_tiles> <rounds> <seed>
 * stdout (one line): COLD fmt=.. VLEN=.. K=.. nr=.. nc=.. pool=.. ws_bytes=.. wbytes_per_call=..
 *   ours_ns_med=.. ours_iqrpct=.. ours_gmacs=.. ours_wGBs=.. opp_ns_med=.. opp_iqrpct=.. opp_gmacs=..
 *   opp_wGBs=.. ratio_ours_over_opp=.. sink=..
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;          /* 144 */
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K; /* 176 */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;                             /* 292 */
_Static_assert(sizeof(block_q4_K)==144,"q4_K");
_Static_assert(sizeof(block_q5_K)==176,"q5_K");
_Static_assert(sizeof(block_q8_K)==292,"q8_K");

extern void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern void tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern void ggml_vec_dot_q4_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
extern void ggml_vec_dot_q5_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p, size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){
  qsort(v,n,sizeof(double),cmp_d);
  *med = (n&1)? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
  double q1=v[n/4], q3=v[(3*n)/4];
  *iqrpct = (*med>0)? 100.0*(q3-q1)/(*med) : 0.0;
}

int main(int argc, char** argv){
    if(argc<8){ fprintf(stderr,"usage: %s q4_K|q5_K K nr nc pool_tiles rounds seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]), nr=atoi(argv[3]), nc=atoi(argv[4]), P=atoi(argv[5]), rounds=atoi(argv[6]);
    rng=(uint64_t)strtoull(argv[7],0,0)|1ull;
    int is_q5 = !strcmp(fmt,"q5_K");
    if(!is_q5 && strcmp(fmt,"q4_K")){ fprintf(stderr,"fmt must be q4_K or q5_K\n"); return 2; }
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8;
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wstride = is_q5 ? 2816 : 2304;
    size_t astride = 1168;
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;
    size_t wblk = is_q5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    size_t wobytes=(size_t)nc*nb*wblk;
    const uint16_t H = 0x2C00;   /* fp16 0.0625 */

    /* single reused activation + output (small; models decode: warm act, cold weights) */
    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    block_q8_K* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    /* POOL of P independent weight tiles (the big streamed operand) */
    uint8_t** Wr=malloc(P*sizeof(uint8_t*));
    uint8_t** Wo=malloc(P*sizeof(uint8_t*));
    if(!Ar||!Or||!Ao||!Oo||!Wr||!Wo){ fprintf(stderr,"OOM meta\n"); return 3; }
    fill_rand(Ar,abytes);
    fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_K));
    for(int g=0;g<grp_r;g++) for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride); for(int j=0;j<4;j++) p[j]=0.01f; }
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    for(int p=0;p<P;p++){
        Wr[p]=aligned_alloc(64,wbytes);
        Wo[p]=aligned_alloc(64,wobytes);
        if(!Wr[p]||!Wo[p]){ fprintf(stderr,"OOM pool %d\n",p); return 3; }
        fill_rand(Wr[p],wbytes); fill_rand(Wo[p],wobytes);
        for(int g=0;g<grp_c;g++) for(int l=0;l<nb;l++){ uint16_t* q=(uint16_t*)(Wr[p]+((size_t)g*nb+l)*wstride); for(int j=0;j<32;j++) q[j]=H; }
        for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* q=(uint16_t*)(Wo[p]+i*wblk); q[0]=H; q[1]=H; }
    }
    size_t ws_ours = (size_t)P*wbytes + abytes + (size_t)nr*nc*sizeof(float);

    #define OURS_CALL(P_IDX) do{ if(is_q5) tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K((size_t)K,Or,Wr[P_IDX],Ar,(size_t)nr,(size_t)nc,(size_t)nc); \
        else tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Or,Wr[P_IDX],Ar,(size_t)nr,(size_t)nc,(size_t)nc); }while(0)
    #define OPP_CALL(P_IDX) do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo[P_IDX]+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; \
            if(is_q5) ggml_vec_dot_q5_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); \
            else      ggml_vec_dot_q4_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)

    /* warmup: 2 full pool sweeps each side (bring code/TLB steady; data stays cold since pool>>LLC) */
    volatile double sink=0;
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){ OURS_CALL(p); } sink+=Or[0]; }
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){ OPP_CALL(p); } sink+=Oo[0]; }

    double *to=malloc(rounds*sizeof(double));
    double *tp=malloc(rounds*sizeof(double));
    for(int rd=0; rd<rounds; rd++){
        double t0=now_ns();
        for(int p=0;p<P;p++){ OURS_CALL(p); }
        double t1=now_ns();
        to[rd]=(t1-t0)/(double)P;         /* ns per cold call */
        sink+=Or[0]+Or[(size_t)nr*nc-1];
        double t2=now_ns();
        for(int p=0;p<P;p++){ OPP_CALL(p); }
        double t3=now_ns();
        tp[rd]=(t3-t2)/(double)P;
        sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
    double macs=(double)nr*(double)nc*(double)K;
    double ours_g=macs/om, opp_g=macs/pm;
    double ours_wGBs=(double)wobytes/om, opp_wGBs=(double)wobytes/pm;  /* weight bytes/ns == GB/s (plain block footprint, comparable both sides) */
    printf("COLD fmt=%s VLEN=%ld K=%d nr=%d nc=%d pool=%d ws_bytes=%zu wbytes_per_call=%zu "
           "ours_ns_med=%.1f ours_iqrpct=%.2f ours_gmacs=%.4f ours_wGBs=%.3f "
           "opp_ns_med=%.1f opp_iqrpct=%.2f opp_gmacs=%.4f opp_wGBs=%.3f "
           "ratio_ours_over_opp=%.4f sink=%.1f\n",
           fmt,vlen_bits,K,nr,nc,P,ws_ours,wobytes,
           om,oi,ours_g,ours_wGBs, pm,pi,opp_g,opp_wGBs, ours_g/opp_g, (double)sink);
    return 0;
}
