/* dequant_flat5_driver.c — A2-batch3 ⑤: FLAT-5 dequantize_row streaming cold-start [DEQ-AXIS].
 *
 * Streaming throughput A/B: OUR weft-emitted dequantize_row (typed_dequantize_row_loop_body)
 * vs the OPPONENT's real as-shipped ggml dequantize_row_<fmt>, linked from stock libggml-base.so.
 * Both read IDENTICAL plain quant block bytes and write K f32 => fair same-footprint mem-bound A/B.
 * Decode is DATA-INDEPENDENT => bounded (&0x3F) fill gives valid steady-state timing AND doubles as
 * the byte-exact certificate: require ours[k]==opp[k] bit-for-bit vs the separately-compiled stock
 * ggml oracle (ZERO-MODEL). Any nonzero mismatch/ULP = genuine weft-vs-ggml decode divergence.
 * FLAT-5: q4_0 q4_1 q5_0 q5_1 q8_0 (WBLK 18/20/22/24/34, qk=32). [NG-4] kernel-axis, NOT e2e.
 *
 * argv: <fmt> <K(mult 32)> <hot_iters> <cold_reps> <seed> [verify_only=0]
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define DQ(f) extern void weft_emitc_dequant_##f##_kernel_dequant_##f(size_t,const uint8_t*,float*);
DQ(q4_0) DQ(q4_1) DQ(q5_0) DQ(q5_1) DQ(q8_0)
#undef DQ
#define OQ(f) extern void dequantize_row_##f(const void*,float*,int64_t);
OQ(q4_0) OQ(q4_1) OQ(q5_0) OQ(q5_1) OQ(q8_0)
#undef OQ

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

/* 224 MiB stream (> rvv 64MiB L3 x3, >> k1 512K L2) guarantees cold DRAM */
#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

enum { D_q40=0,D_q41,D_q50,D_q51,D_q80, NF };
static const char* FNAME[NF]={"q4_0","q4_1","q5_0","q5_1","q8_0"};
static const int WBLK[NF]={18,20,22,24,34};
static const int QK[NF]={32,32,32,32,32};

static void our_call(int fmt,size_t k,const uint8_t*x,float*y){ switch(fmt){
    case D_q40: weft_emitc_dequant_q4_0_kernel_dequant_q4_0(k,x,y); break;
    case D_q41: weft_emitc_dequant_q4_1_kernel_dequant_q4_1(k,x,y); break;
    case D_q50: weft_emitc_dequant_q5_0_kernel_dequant_q5_0(k,x,y); break;
    case D_q51: weft_emitc_dequant_q5_1_kernel_dequant_q5_1(k,x,y); break;
    default:    weft_emitc_dequant_q8_0_kernel_dequant_q8_0(k,x,y); } }
static void opp_call(int fmt,const uint8_t*x,float*y,int64_t k){ switch(fmt){
    case D_q40: dequantize_row_q4_0(x,y,k); break;
    case D_q41: dequantize_row_q4_1(x,y,k); break;
    case D_q50: dequantize_row_q5_0(x,y,k); break;
    case D_q51: dequantize_row_q5_1(x,y,k); break;
    default:    dequantize_row_q8_0(x,y,k); } }

int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: %s fmt K hot_iters cold_reps seed [verify_only]\n",argv[0]); return 2; }
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

    size_t wbytes=(size_t)nb*WBLK[fmt];
    uint8_t* W=aligned_alloc(64,wbytes);
    float* Our=aligned_alloc(64,(size_t)K*sizeof(float));
    float* Opp=aligned_alloc(64,(size_t)K*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!W||!Our||!Opp||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    for(size_t i=0;i<wbytes;i++) W[i]=(uint8_t)(xr()&0x3F);
    memset(g_flush,1,FLUSH_BYTES);

    our_call(fmt,(size_t)K,W,Our); opp_call(fmt,W,Opp,(int64_t)K);
    long long mism=0; int64_t worst_ulp=0; int firsti=-1;
    for(size_t k=0;k<(size_t)K;k++){
        float a=Our[k], b=Opp[k];
        if(memcmp(&a,&b,4)!=0){ mism++; if(firsti<0)firsti=(int)k; }
        uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4);
        int64_t ka=(ua&0x80000000u)?-(int64_t)(ua&0x7fffffffu):(int64_t)ua;
        int64_t kb=(ub&0x80000000u)?-(int64_t)(ub&0x7fffffffu):(int64_t)ub;
        int64_t u=ka-kb; if(u<0)u=-u; if(u>worst_ulp)worst_ulp=u;
    }
    printf("VERIFY fmt=%s K=%d : byte_mismatch=%lld worst_ulp=%lld first_i=%d sample_our=%.6f opp=%.6f => %s\n",
           FNAME[fmt],K,mism,(long long)worst_ulp,firsti,Our[0],Opp[0], mism? "*** MISMATCH ***":"BYTE-EXACT-vs-stock-ggml OK");
    if(vonly){ free(W);free(Our);free(Opp);free(g_flush); return mism?1:0; }

    double elems=(double)K;
    double in_bytes=(double)wbytes, out_bytes=(double)K*4.0;

    for(int w=0;w<3;w++){ our_call(fmt,(size_t)K,W,Our); }
    double ours_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) our_call(fmt,(size_t)K,W,Our); double npc=(now_ns()-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; }
    for(int w=0;w<2;w++){ opp_call(fmt,W,Opp,(int64_t)K); }
    double opp_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) opp_call(fmt,W,Opp,(int64_t)K); double npc=(now_ns()-t0)/(double)hiters; if(npc<opp_hot)opp_hot=npc; }

    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    cold_flush(); our_call(fmt,(size_t)K,W,Our); cold_flush(); opp_call(fmt,W,Opp,(int64_t)K);
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); our_call(fmt,(size_t)K,W,Our); ours[p]=now_ns()-t0;
        cold_flush(); double t1=now_ns(); opp_call(fmt,W,Opp,(int64_t)K); opp[p] =now_ns()-t1;
    }
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }
    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    double o_q1=ours[reps/4], o_q3=ours[(3*reps)/4], p_q1=opp[reps/4], p_q3=opp[(3*reps)/4];
    double o_iqr = om>0?100.0*(o_q3-o_q1)/om:0.0, p_iqr = pm>0?100.0*(p_q3-p_q1)/pm:0.0;
    double our_gbs = (in_bytes+out_bytes)/om, opp_gbs=(in_bytes+out_bytes)/pm;

    printf("CENSUS fmt=%s VLEN=%ld K=%d qk=%d WBLK=%d hiters=%d reps=%d nf_ns=%.0f | "
           "HOT ours_ns=%.0f opp_ns=%.0f ratio_hot=%.4f | "
           "COLD ours_med_ns=%.0f o_iqr=%.2f opp_med_ns=%.0f p_iqr=%.2f ratio_cold_med=%.4f ratio_cold_best=%.4f "
           "our_GBs=%.2f opp_GBs=%.2f | byte_mismatch=%lld worst_ulp=%lld\n",
           FNAME[fmt],vlen,K,qk,WBLK[fmt],hiters,reps,nf,
           ours_hot,opp_hot,(elems/ours_hot)/(elems/opp_hot),
           om,o_iqr,pm,p_iqr,(elems/om)/(elems/pm),(elems/ob)/(elems/pb),our_gbs,opp_gbs,
           mism,(long long)worst_ulp);
    free(W);free(Our);free(Opp);free(g_flush);free(ours);free(opp);
    return mism?1:0;
}
