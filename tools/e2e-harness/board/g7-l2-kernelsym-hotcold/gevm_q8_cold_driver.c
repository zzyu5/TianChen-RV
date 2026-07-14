/* gevm_q8_cold_driver.c — G7 L2 货架A hot/cold micro A/B for q8_0 GEVM (decode, M=1) @k1/VLEN256.
 *
 * q8_0 GEVM is the DECODE-native regime (nr=1, one token). Dequant is trivial (int8*int8 -> i32,
 * one f16 scale), so unlike K-quant this op can actually approach the DRAM memory wall — making
 * its cold micro the best candidate "e2e-decode predictor". This driver times OUR deployed wide
 * strip-width GEVM (block_q8_0x16 repack) vs the board's own ggml_vec_dot_q8_0_q8_0 block-dot
 * (factory-as-shipped, plain q8_0 blocks) under BOTH:
 *   HOT  = single weight tile, warmup + best-of-5 of `iters` back-to-back calls (cache-warm-ish;
 *          note k1 L2=512KiB — one tile at nc>=2048 exceeds L2 so even hot is only partly warm).
 *   COLD = POOL of P independent weight tiles (combined >> LLC); each round sweeps the whole pool
 *          once (one call/tile), median-of-N. Every weight tile read cold from DRAM = decode model.
 * Reports achieved weight-GB/s so we can see whether q8_0 GEVM reaches the memory wall (unlike
 * the compute-bound K-quant repack GEMM).
 *
 * Opponent = board libggml-cpu.so ggml_vec_dot_q8_0_q8_0 (block-dot, memory-bound weak opponent per
 * T9). [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win. Ours=clang-18; opp=factory gcc-15.
 * Correctness: this is a throughput probe (data-independent control flow); q8_0 repack byte-exactness
 * sealed elsewhere. Weight bytes plain(nc*nb*34)==repacked(grp_c*nb*544) identical footprint both sides.
 *
 * argv: <K(mult32)> <nc(mult16)> <hot_iters> <pool_tiles> <rounds> <seed>
 * stdout: two lines HOT..  and  COLD..  (parseable).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0; /* 34 */
_Static_assert(sizeof(block_q8_0)==34,"q8_0");

extern void tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0(
    size_t n, float*s, const uint8_t*vx, const uint8_t*vy, size_t nc);
extern void ggml_vec_dot_q8_0_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  ri8(void){ return (int8_t)((int)(xr()%255)-127); }
static uint16_t rf16(void){ uint16_t s=(uint16_t)((xr()&1)<<15),e=(uint16_t)((10+(xr()%8))&0x1F),m=(uint16_t)(xr()&0x3FF); return (uint16_t)(s|(e<<10)|m); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){ qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]); double q1=v[n/4],q3=v[(3*n)/4]; *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0; }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s K nc hot_iters pool_tiles rounds seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nc=atoi(argv[2]), hiters=atoi(argv[3]), P=atoi(argv[4]), rounds=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(K%32||nc%16){ fprintf(stderr,"K%%32,nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8; if(hiters<1)hiters=1;
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK8_0, grp_c=nc/16;
    size_t wr_bytes=(size_t)grp_c*nb*544;         /* ours repacked block_q8_0x16 */
    size_t wo_bytes=(size_t)nc*nb*sizeof(block_q8_0); /* opp plain col-major */
    size_t a_bytes=(size_t)nb*34;

    /* single reused activation (warm current token) + outputs */
    uint8_t* ra=malloc(a_bytes);
    block_q8_0* Ao=malloc((size_t)nb*sizeof(block_q8_0));
    float* out_o=malloc((size_t)nc*sizeof(float));
    float* out_p=malloc((size_t)nc*sizeof(float));
    uint8_t** Wr=malloc(P*sizeof(uint8_t*));      /* ours repacked weight pool */
    block_q8_0** Wo=malloc(P*sizeof(block_q8_0*)); /* opp plain weight pool */
    if(!ra||!Ao||!out_o||!out_p||!Wr||!Wo){ fprintf(stderr,"OOM meta\n"); return 3; }
    for(int b=0;b<nb;b++){ Ao[b].d=rf16(); for(int j=0;j<QK8_0;j++) Ao[b].qs[j]=ri8(); }
    for(int b=0;b<nb;b++){ uint8_t*blk=ra+(size_t)b*34; memcpy(blk,&Ao[b].d,2); for(int l=0;l<QK8_0;l++) blk[2+l]=(uint8_t)Ao[b].qs[l]; }

    for(int p=0;p<P;p++){
        Wr[p]=malloc(wr_bytes);
        Wo[p]=malloc((size_t)nc*nb*sizeof(block_q8_0));
        if(!Wr[p]||!Wo[p]){ fprintf(stderr,"OOM pool %d\n",p); return 3; }
        for(int c=0;c<nc;c++) for(int b=0;b<nb;b++){ Wo[p][c*nb+b].d=rf16(); for(int j=0;j<QK8_0;j++) Wo[p][c*nb+b].qs[j]=ri8(); }
        for(int g=0;g<grp_c;g++) for(int b=0;b<nb;b++){ uint8_t*blk=Wr[p]+((size_t)g*nb+b)*544;
            for(int cc=0;cc<16;cc++){ int c=g*16+cc; memcpy(blk+cc*2,&Wo[p][c*nb+b].d,2); }
            for(int l=0;l<QK8_0;l++) for(int cc=0;cc<16;cc++){ int c=g*16+cc; blk[32+l*16+cc]=(uint8_t)Wo[p][c*nb+b].qs[l]; } }
    }
    #define OURS(PI) tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0((size_t)K,out_o,Wr[PI],ra,(size_t)nc)
    #define OPP(PI)  do{ for(int c=0;c<nc;c++) ggml_vec_dot_q8_0_q8_0(K,&out_p[c],0,&Wo[PI][(size_t)c*nb],0,Ao,0,1); }while(0)
    volatile double sink=0;
    double macs=(double)nc*(double)K;

    /* ---- HOT: single tile 0, warmup + best-of-5 of hiters back-to-back ---- */
    for(int w=0;w<3;w++){ OURS(0); OPP(0); }
    double ours_hot=1e30, opp_hot=1e30;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns(); for(int it=0;it<hiters;it++){ OURS(0); } double t1=now_ns();
        double npc=(t1-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; sink+=out_o[0];
        double t2=now_ns(); for(int it=0;it<hiters;it++){ OPP(0); } double t3=now_ns();
        npc=(t3-t2)/(double)hiters; if(npc<opp_hot)opp_hot=npc; sink+=out_p[0];
    }
    printf("HOT fmt=q8_0 VLEN=%ld K=%d nc=%d hiters=%d wbytes_per_call=%zu "
           "ours_ns=%.1f ours_gmacs=%.4f ours_wGBs=%.3f opp_ns=%.1f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           vlen_bits,K,nc,hiters,wo_bytes, ours_hot,macs/ours_hot,(double)wo_bytes/ours_hot,
           opp_hot,macs/opp_hot,(double)wo_bytes/opp_hot, (macs/ours_hot)/(macs/opp_hot),(double)sink);

    /* ---- COLD: pool sweep, median-of-N ---- */
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){ OURS(p); } for(int p=0;p<P;p++){ OPP(p); } sink+=out_o[0]+out_p[0]; }
    double *to=malloc(rounds*sizeof(double)), *tp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++){ OURS(p); } double t1=now_ns(); to[rd]=(t1-t0)/(double)P; sink+=out_o[0];
        double t2=now_ns(); for(int p=0;p<P;p++){ OPP(p); }  double t3=now_ns(); tp[rd]=(t3-t2)/(double)P; sink+=out_p[0];
    }
    double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
    size_t ws=(size_t)P*wo_bytes;
    printf("COLD fmt=q8_0 VLEN=%ld K=%d nc=%d pool=%d ws_bytes=%zu wbytes_per_call=%zu "
           "ours_ns_med=%.1f ours_iqrpct=%.2f ours_gmacs=%.4f ours_wGBs=%.3f "
           "opp_ns_med=%.1f opp_iqrpct=%.2f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           vlen_bits,K,nc,P,ws,wo_bytes, om,oi,macs/om,(double)wo_bytes/om,
           pm,pi,macs/pm,(double)wo_bytes/pm, (macs/om)/(macs/pm),(double)sink);
    return 0;
}
