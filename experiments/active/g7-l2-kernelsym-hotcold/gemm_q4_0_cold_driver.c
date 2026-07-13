/* gemm_q4_0_cold_driver.c — G7 L2 货架A hot/cold micro A/B for q4_0 repack GEMM (prefill) @k1/VLEN256.
 *
 * Times OUR front-door q4_0 16x1 repack GEMM vs board ggml_vec_dot_q4_0_q8_0 block-dot (factory
 * as-shipped, plain blocks). HOT = single tile best-of-5; COLD = pool>>LLC median-of-N (weights
 * read cold from DRAM). nr is a CLI arg so nr=64 (prefill/hot-shape) and nr=4 (decode-leaning min).
 * Reports achieved weight-GB/s for roofline classification.
 *
 * NB the sealed hot q4_0@k1 number (1.0022x) was a VLEN-flip prefill *control* (self); this driver
 * instead pairs against factory block-dot for an honest weak-opponent cold ratio. [NG-4] kernel-axis
 * datapoint, NOT e2e, NOT a sealed Win. Ours=clang-18; opp=factory gcc-15. Throughput probe
 * (data-independent control flow); q4_0 repack byte-exactness sealed elsewhere.
 *
 * argv: <K(mult32)> <nr(mult4)> <nc(mult16)> <hot_iters> <pool_tiles> <rounds> <seed>
 * stdout: HOT.. and COLD.. lines (parseable).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
#define QK4_0 32
#define QK8_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0; /* 18 */
typedef struct { ggml_half d; int8_t  qs[QK8_0];    } block_q8_0; /* 34 */
_Static_assert(sizeof(block_q4_0)==18,"q4_0");
_Static_assert(sizeof(block_q8_0)==34,"q8_0");

extern void tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
    size_t nr, size_t bs, size_t n, float*s, size_t nc, const uint8_t*vx, const uint8_t*vy);
extern void ggml_vec_dot_q4_0_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  ri8(void){ return (int8_t)((int)(xr()%255)-127); }
static uint8_t ru8(void){ return (uint8_t)(xr()&0xff); }
static uint16_t rf16(void){ uint16_t s=(uint16_t)((xr()&1)<<15),e=(uint16_t)((10+(xr()%8))&0x1F),m=(uint16_t)(xr()&0x3FF); return (uint16_t)(s|(e<<10)|m); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){ qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]); double q1=v[n/4],q3=v[(3*n)/4]; *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0; }

int main(int argc,char**argv){
    if(argc<8){ fprintf(stderr,"usage: %s K nr nc hot_iters pool_tiles rounds seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]), hiters=atoi(argv[4]), P=atoi(argv[5]), rounds=atoi(argv[6]);
    rng=(uint64_t)strtoull(argv[7],0,0)|1ull;
    if(K%32||nr%4||nc%16){ fprintf(stderr,"K%%32,nr%%4,nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8; if(hiters<1)hiters=1;
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK4_0, grp_c=nc/16, grp_r=nr/4;
    size_t wr_bytes=(size_t)grp_c*nb*288, ar_bytes=(size_t)grp_r*nb*136;
    size_t wo_bytes=(size_t)nc*nb*sizeof(block_q4_0);        /* plain weight footprint (both sides same total) */

    uint8_t* ra=malloc(ar_bytes);
    block_q8_0* Ao=malloc((size_t)nr*nb*sizeof(block_q8_0));
    float* out_o=malloc((size_t)nr*nc*sizeof(float));
    float* out_p=malloc((size_t)nr*nc*sizeof(float));
    uint8_t** Wr=malloc(P*sizeof(uint8_t*));
    block_q4_0** Wo=malloc(P*sizeof(block_q4_0*));
    if(!ra||!Ao||!out_o||!out_p||!Wr||!Wo){ fprintf(stderr,"OOM meta\n"); return 3; }
    for(int r=0;r<nr;r++) for(int b=0;b<nb;b++){ Ao[r*nb+b].d=rf16(); for(int j=0;j<QK8_0;j++) Ao[r*nb+b].qs[j]=ri8(); }
    /* repack activations block_q8_0x4 (stride 136) */
    for(int g=0;g<grp_r;g++) for(int b=0;b<nb;b++){ uint8_t*blk=ra+((size_t)g*nb+b)*136;
        for(int rr=0;rr<4;rr++){ int r=g*4+rr; memcpy(blk+rr*2,&Ao[r*nb+b].d,2); }
        for(int l=0;l<QK8_0/2;l++) for(int rr=0;rr<4;rr++){ int r=g*4+rr;
            blk[8+l*4+rr]=(uint8_t)Ao[r*nb+b].qs[l]; blk[72+l*4+rr]=(uint8_t)Ao[r*nb+b].qs[l+QK8_0/2]; } }
    for(int p=0;p<P;p++){
        Wr[p]=malloc(wr_bytes);
        Wo[p]=malloc((size_t)nc*nb*sizeof(block_q4_0));
        if(!Wr[p]||!Wo[p]){ fprintf(stderr,"OOM pool %d\n",p); return 3; }
        for(int c=0;c<nc;c++) for(int b=0;b<nb;b++){ Wo[p][c*nb+b].d=rf16(); for(int j=0;j<QK4_0/2;j++) Wo[p][c*nb+b].qs[j]=ru8(); }
        for(int g=0;g<grp_c;g++) for(int b=0;b<nb;b++){ uint8_t*blk=Wr[p]+((size_t)g*nb+b)*288;
            for(int cc=0;cc<16;cc++){ int c=g*16+cc; memcpy(blk+cc*2,&Wo[p][c*nb+b].d,2); }
            for(int l=0;l<QK4_0/2;l++) for(int cc=0;cc<16;cc++){ int c=g*16+cc; blk[32+l*16+cc]=(uint8_t)(Wo[p][c*nb+b].qs[l]^0x88); } }
    }
    #define OURS(PI) tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm((size_t)nr,(size_t)nc,(size_t)K,out_o,(size_t)nc,Wr[PI],ra)
    #define OPP(PI)  do{ for(int c=0;c<nc;c++){ const block_q4_0* wc=&Wo[PI][(size_t)c*nb]; \
        for(int r=0;r<nr;r++){ const block_q8_0* ar=&Ao[(size_t)r*nb]; ggml_vec_dot_q4_0_q8_0(K,&out_p[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)
    volatile double sink=0; double macs=(double)nr*(double)nc*(double)K;

    for(int w=0;w<3;w++){ OURS(0); OPP(0); }
    double ours_hot=1e30, opp_hot=1e30;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns(); for(int it=0;it<hiters;it++){ OURS(0); } double t1=now_ns();
        double npc=(t1-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; sink+=out_o[0];
        double t2=now_ns(); for(int it=0;it<hiters;it++){ OPP(0); } double t3=now_ns();
        npc=(t3-t2)/(double)hiters; if(npc<opp_hot)opp_hot=npc; sink+=out_p[0];
    }
    printf("HOT fmt=q4_0 VLEN=%ld K=%d nr=%d nc=%d hiters=%d wbytes_per_call=%zu "
           "ours_ns=%.1f ours_gmacs=%.4f ours_wGBs=%.3f opp_ns=%.1f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           vlen_bits,K,nr,nc,hiters,wo_bytes, ours_hot,macs/ours_hot,(double)wo_bytes/ours_hot,
           opp_hot,macs/opp_hot,(double)wo_bytes/opp_hot,(macs/ours_hot)/(macs/opp_hot),(double)sink);

    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){ OURS(p); } for(int p=0;p<P;p++){ OPP(p); } sink+=out_o[0]+out_p[0]; }
    double *to=malloc(rounds*sizeof(double)), *tp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++){ OURS(p); } double t1=now_ns(); to[rd]=(t1-t0)/(double)P; sink+=out_o[0];
        double t2=now_ns(); for(int p=0;p<P;p++){ OPP(p); }  double t3=now_ns(); tp[rd]=(t3-t2)/(double)P; sink+=out_p[0];
    }
    double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
    printf("COLD fmt=q4_0 VLEN=%ld K=%d nr=%d nc=%d pool=%d ws_bytes=%zu wbytes_per_call=%zu "
           "ours_ns_med=%.1f ours_iqrpct=%.2f ours_gmacs=%.4f ours_wGBs=%.3f "
           "opp_ns_med=%.1f opp_iqrpct=%.2f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           vlen_bits,K,nr,nc,P,(size_t)P*wo_bytes,wo_bytes, om,oi,macs/om,(double)wo_bytes/om,
           pm,pi,macs/pm,(double)wo_bytes/pm,(macs/om)/(macs/pm),(double)sink);
    return 0;
}
