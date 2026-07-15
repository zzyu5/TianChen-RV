/* quantize3_driver.c — A2-batch3 ③: quantize_row_q8_{0,1,K} streaming cold-start A/B (VECTOR fight).
 *
 * Streaming throughput A/B: OUR weft-emitted quantize_row (typed_quantize_row_loop_body, ggml's exact
 * RVV method: vfredmax amax + f32->i16->i8 RNE narrow + int8 store) vs the OPPONENT's real as-shipped
 * ggml quantize_row_q8_<fmt> (arch/riscv/quants.c hand-written __riscv_v intrinsic), linked from stock
 * libggml-cpu.so. Both read IDENTICAL f32 input and write quant blocks => fair same-footprint A/B.
 * Quantize is DATA-DEPENDENT (amax per block sets scale) => fill input with real floats; byte-exact
 * ZERO-MODEL certificate = compare OUTPUT quant bytes bit-for-bit vs separately-compiled stock oracle.
 * Any nonzero output mismatch = genuine weft-vs-ggml quantize divergence (reported honestly).
 * q8_0 stride=34/qk=32, q8_1 stride=36/qk=32, q8_K stride=292/qk=256. [NG-4] kernel-axis, NOT e2e.
 *
 * argv: <fmt> <K(mult qk)> <hot_iters> <cold_reps> <seed> [verify_only=0]
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QZ(f) extern void weft_emitc_quantize_row_##f##_kernel_quantize_row_##f(size_t,const float*,uint8_t*);
QZ(q8_0) QZ(q8_1) QZ(q8_K)
#undef QZ
#define OQ(f) extern void quantize_row_##f(const float*,void*,int64_t);
OQ(q8_0) OQ(q8_1) OQ(q8_K)
#undef OQ

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float frand(void){ /* ~[-8,8) f32 */ return ((float)(xr()>>8)/(float)(1u<<23))*16.0f - 8.0f; }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

enum { Q80=0,Q81,Q8K, NF };
static const char* FNAME[NF]={"q8_0","q8_1","q8_K"};
static const int STRIDE[NF]={34,36,292};
static const int QK[NF]={32,32,256};

static void our_call(int fmt,size_t k,const float*x,uint8_t*y){ switch(fmt){
    case Q80: weft_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(k,x,y); break;
    case Q81: weft_emitc_quantize_row_q8_1_kernel_quantize_row_q8_1(k,x,y); break;
    default:  weft_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(k,x,y); } }
static void opp_call(int fmt,const float*x,uint8_t*y,int64_t k){ switch(fmt){
    case Q80: quantize_row_q8_0(x,y,k); break;
    case Q81: quantize_row_q8_1(x,y,k); break;
    default:  quantize_row_q8_K(x,y,k); } }

int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: %s fmt K hot cold seed [verify_only]\n",argv[0]); return 2; }
    const char* fs=argv[1];
    int fmt=-1; for(int i=0;i<NF;i++) if(!strcmp(fs,FNAME[i])) fmt=i;
    if(fmt<0){ fprintf(stderr,"bad fmt %s\n",fs); return 2; }
    int K=atoi(argv[2]),hiters=atoi(argv[3]),reps=atoi(argv[4]);
    rng=(uint64_t)strtoull(argv[5],0,0)|1ull;
    int vonly = argc>6?atoi(argv[6]):0;
    int qk=QK[fmt];
    if(K%qk){ fprintf(stderr,"K must be mult of %d\n",qk); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    int nb=K/qk;
    long vlen=(long)__riscv_vlenb()*8;

    size_t obytes=(size_t)nb*STRIDE[fmt];
    float* X=aligned_alloc(64,(size_t)K*sizeof(float));
    uint8_t* Our=aligned_alloc(64,obytes+64);
    uint8_t* Opp=aligned_alloc(64,obytes+64);
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!X||!Our||!Opp||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    for(size_t i=0;i<(size_t)K;i++) X[i]=frand();
    memset(g_flush,1,FLUSH_BYTES);

    /* ===== VERIFY (byte-exact output vs stock ggml oracle) ===== */
    memset(Our,0xAA,obytes); memset(Opp,0x55,obytes);
    our_call(fmt,(size_t)K,X,Our); opp_call(fmt,X,Opp,(int64_t)K);
    long long mism=0; int firsti=-1;
    for(size_t i=0;i<obytes;i++){ if(Our[i]!=Opp[i]){ mism++; if(firsti<0)firsti=(int)i; } }
    printf("VERIFY fmt=%s K=%d obytes=%zu : out_byte_mismatch=%lld first_i=%d => %s\n",
           FNAME[fmt],K,obytes,mism,firsti, mism? "*** MISMATCH ***":"BYTE-EXACT-vs-stock-ggml OK");
    if(vonly){ free(X);free(Our);free(Opp);free(g_flush); return mism?1:0; }

    double elems=(double)K;
    double in_bytes=(double)K*4.0, out_bytes=(double)obytes;

    for(int w=0;w<3;w++){ our_call(fmt,(size_t)K,X,Our); }
    double ours_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) our_call(fmt,(size_t)K,X,Our); double npc=(now_ns()-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; }
    for(int w=0;w<2;w++){ opp_call(fmt,X,Opp,(int64_t)K); }
    double opp_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) opp_call(fmt,X,Opp,(int64_t)K); double npc=(now_ns()-t0)/(double)hiters; if(npc<opp_hot)opp_hot=npc; }

    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    cold_flush(); our_call(fmt,(size_t)K,X,Our); cold_flush(); opp_call(fmt,X,Opp,(int64_t)K);
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); our_call(fmt,(size_t)K,X,Our); ours[p]=now_ns()-t0;
        cold_flush(); double t1=now_ns(); opp_call(fmt,X,Opp,(int64_t)K); opp[p] =now_ns()-t1;
    }
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }
    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    double o_q1=ours[reps/4], o_q3=ours[(3*reps)/4], p_q1=opp[reps/4], p_q3=opp[(3*reps)/4];
    double o_iqr = om>0?100.0*(o_q3-o_q1)/om:0.0, p_iqr = pm>0?100.0*(p_q3-p_q1)/pm:0.0;
    double our_gbs = (in_bytes+out_bytes)/om, opp_gbs=(in_bytes+out_bytes)/pm;

    printf("QUANT fmt=%s VLEN=%ld K=%d qk=%d STRIDE=%d hiters=%d reps=%d nf_ns=%.0f | "
           "HOT ours_ns=%.0f opp_ns=%.0f ratio_hot=%.4f | "
           "COLD ours_med_ns=%.0f o_iqr=%.2f opp_med_ns=%.0f p_iqr=%.2f ratio_cold_med=%.4f ratio_cold_best=%.4f "
           "our_GBs=%.2f opp_GBs=%.2f | out_byte_mismatch=%lld\n",
           FNAME[fmt],vlen,K,qk,STRIDE[fmt],hiters,reps,nf,
           ours_hot,opp_hot,(elems/ours_hot)/(elems/opp_hot),
           om,o_iqr,pm,p_iqr,(elems/om)/(elems/pm),(elems/ob)/(elems/pb),our_gbs,opp_gbs,mism);
    free(X);free(Our);free(Opp);free(g_flush);free(ours);free(opp);
    return mism?1:0;
}
