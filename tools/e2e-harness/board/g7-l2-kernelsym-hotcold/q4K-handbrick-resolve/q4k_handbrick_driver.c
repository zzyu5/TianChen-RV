/* q4k_handbrick_driver.c — G7 货架A · T9 §6 争议解: q4_K@k1 our-emit repack-GEMM
 * vs the TRUE shipped hand-brick ggml_gemm_q4_K_16x1_q8_K (case256 VLEN256-native repack).
 *
 * BOTH kernels consume the SAME repacked weight buffer (block_q4_Kx16, 2304B/col-group-superblock)
 * and SAME q8 activation (blk_q8_Kx4, 1168B/row-group) — this is the byte-layout drop-in established
 * by t4b-m4-decisive (m4_probe fed identical vx/vy to ours + ggml_gemm_q4_K_16x1_q8_K). We add:
 *   [A] correctness cross-check (ours vs 16x1 over ALL nr*nc; detects partial-row calling bug)
 *   [B] HOT timing (single buffer, median-of-N rounds, iters inner)  — paired within-process
 *   [C] COLD timing (P-tile pool >> LLC, median-of-N rounds)         — paired within-process
 *
 * Opponent = ggml_gemm_q4_K_16x1_q8_K from /data/k1build-stock libggml-cpu.so (clang++-18 -O3,
 * arch/riscv/repack.cpp — verified via compile_commands.json). Ours = clang-18 (same compiler).
 * K-quant decode has NO data branches => throughput is data-independent; random fills are valid for
 * the timing probe, and the correctness check confirms both compute the same GEMM at VLEN256.
 * [NG-4]: kernel-axis micro A/B, NOT an e2e beat, NOT a sealed 8-gate Win.
 *
 * argv: <K(mult256)> <nr(mult4)> <nc(mult16)> <pool_tiles> <rounds> <iters_hot> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;

/* ours: (n, s, vx, vy, nr, nc, bs) */
extern void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
/* true shipped hand-brick: (n, s, bs, vx, vy, nr, nc) */
extern void ggml_gemm_q4_K_16x1_q8_K(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);

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

/* fill one repacked weight superblock-pool tile + activation-consistent scales (H=0.0625, d=0.01) */
static const uint16_t H = 0x2C00;
static void seed_scales(uint8_t* W, size_t wbytes, int grp_c, int nb){
    for(int g=0;g<grp_c;g++) for(int l=0;l<nb;l++){ uint16_t* q=(uint16_t*)(W+((size_t)g*nb+l)*2304); for(int j=0;j<32;j++) q[j]=H; }
    (void)wbytes;
}

int main(int argc, char** argv){
    if(argc<8){ fprintf(stderr,"usage: %s K nr nc pool rounds iters_hot seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]), P=atoi(argv[4]), rounds=atoi(argv[5]), iters=atoi(argv[6]);
    rng=(uint64_t)strtoull(argv[7],0,0)|1ull;
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8; if(iters<1)iters=1;
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    size_t wbytes=(size_t)grp_c*nb*2304;      /* block_q4_Kx16 repacked weight tile */
    size_t abytes=(size_t)grp_r*nb*1168;      /* blk_q8_Kx4 activation */
    size_t obytes=(size_t)nr*nc*sizeof(float);

    uint8_t* A =aligned_alloc(64,abytes);
    float*   Oo=aligned_alloc(64,obytes);     /* opp (16x1) output */
    float*   Or=aligned_alloc(64,obytes);     /* ours output */
    if(!A||!Oo||!Or){ fprintf(stderr,"OOM meta\n"); return 3; }
    fill_rand(A,abytes);
    for(int g=0;g<grp_r;g++) for(int l=0;l<nb;l++){ float* p=(float*)(A+((size_t)g*nb+l)*1168); for(int j=0;j<4;j++) p[j]=0.01f; }

    /* POOL of P independent repacked weight tiles */
    uint8_t** W=malloc(P*sizeof(uint8_t*));
    if(!W){ fprintf(stderr,"OOM Wptr\n"); return 3; }
    for(int p=0;p<P;p++){ W[p]=aligned_alloc(64,wbytes); if(!W[p]){fprintf(stderr,"OOM pool\n");return 3;} fill_rand(W[p],wbytes); seed_scales(W[p],wbytes,grp_c,nb); }

    #define OURS(PI) tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Or,W[PI],A,(size_t)nr,(size_t)nc,(size_t)nc)
    #define OPP(PI)  ggml_gemm_q4_K_16x1_q8_K(K,Oo,(size_t)nc,W[PI],A,nr,nc)

    /* [A] CORRECTNESS: ours vs 16x1 over ALL nr*nc (detect partial-row calling bug) */
    memset(Or,0,obytes); memset(Oo,0,obytes);
    OURS(0); OPP(0);
    double max_abs=0, max_rel=0; int first_bad_row=-1, nbad=0;
    for(int r=0;r<nr;r++){ int rowbad=0;
      for(int c=0;c<nc;c++){ double o=Or[(size_t)r*nc+c], g=Oo[(size_t)r*nc+c];
        double a=fabs(o-g), rel=a/(fabs(g)+1e-6);
        if(a>max_abs)max_abs=a; if(rel>max_rel)max_rel=rel;
        if(a>1e-2*(fabs(g)+1e-3)){ nbad++; rowbad=1; }
      }
      if(rowbad&&first_bad_row<0) first_bad_row=r;
    }
    printf("CORRECTNESS ours_vs_16x1: max_abs=%.3e max_rel=%.3e nbad=%d/%d first_bad_row=%d "
           "ours[0]=%.4f opp[0]=%.4f ours[last]=%.4f opp[last]=%.4f\n",
           max_abs,max_rel,nbad,nr*nc,first_bad_row,
           Or[0],Oo[0],Or[(size_t)nr*nc-1],Oo[(size_t)nr*nc-1]);

    volatile double sink=0;
    double macs=(double)nr*(double)nc*(double)K;

    /* [B] HOT: single tile W[0], each round = median over iters back-to-back calls (best-ish), median-of-rounds */
    for(int w=0;w<3;w++){ OURS(0); OPP(0); sink+=Or[0]+Oo[0]; }
    double *ho=malloc(rounds*sizeof(double)), *hp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int it=0;it<iters;it++){ OURS(0); sink+=Or[0]; } double t1=now_ns();
        ho[rd]=(t1-t0)/(double)iters;
        double t2=now_ns(); for(int it=0;it<iters;it++){ OPP(0); sink+=Oo[0]; } double t3=now_ns();
        hp[rd]=(t3-t2)/(double)iters;
    }
    double hom,hoi,hpm,hpi; stats(ho,rounds,&hom,&hoi); stats(hp,rounds,&hpm,&hpi);
    printf("HOT  VLEN=%ld K=%d nr=%d nc=%d iters=%d rounds=%d "
           "ours_ns=%.1f ours_iqr=%.2f%% ours_gmacs=%.4f opp_ns=%.1f opp_iqr=%.2f%% opp_gmacs=%.4f "
           "ratio_ours_over_opp=%.4f\n",
           vlen_bits,K,nr,nc,iters,rounds, hom,hoi,macs/hom, hpm,hpi,macs/hpm, (macs/hom)/(macs/hpm));

    /* [C] COLD: sweep P-tile pool each round (weights DRAM-cold), median-of-rounds */
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){OURS(p);} sink+=Or[0]; }
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){OPP(p);}  sink+=Oo[0]; }
    double *co=malloc(rounds*sizeof(double)), *cp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++){OURS(p);} double t1=now_ns();
        co[rd]=(t1-t0)/(double)P; sink+=Or[0]+Or[(size_t)nr*nc-1];
        double t2=now_ns(); for(int p=0;p<P;p++){OPP(p);}  double t3=now_ns();
        cp[rd]=(t3-t2)/(double)P; sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }
    double com,coi,cpm,cpi; stats(co,rounds,&com,&coi); stats(cp,rounds,&cpm,&cpi);
    printf("COLD VLEN=%ld K=%d nr=%d nc=%d pool=%d rounds=%d "
           "ours_ns=%.1f ours_iqr=%.2f%% ours_gmacs=%.4f opp_ns=%.1f opp_iqr=%.2f%% opp_gmacs=%.4f "
           "ratio_ours_over_opp=%.4f sink=%.1f\n",
           vlen_bits,K,nr,nc,P,rounds, com,coi,macs/com, cpm,cpi,macs/cpm, (macs/com)/(macs/cpm), (double)sink);
    return 0;
}
