/* preduce_driver.c — K线 product_reduce 对拍/计时 driver（bench cells/product_reduce.sh 的板端载荷）.
 *
 * op = product_reduce（N3 gearbox 归约子原语）。三格：
 *   q4_0_nibble    (unsigned-nibble):  acc += (b&0xF)*a[i] + (b>>4)*c[i]
 *   offset_binary_n3 (q5_0 five-bit offset-binary): 5th bit from qh[2..3]/[4..5], val=(nib|bit<<4)-16
 *   codebook_n3    (iq4_nl codebook-gather): val=kvalues[nib], acc += val_lo*a + val_hi*c
 *
 * ★对手（部署事实·对手法 §3.4）：ggml 无 standalone product_reduce 框架核（内部子原语·
 *   T-CENSUS §一.F）。公平参照 = 计算【同一】整数算术的 SCALAR-REF（标量类档·codegen-lottery，
 *   与 dequantize_row 的标量类对手同性质）。故这是 SANITY 层 A/B（ours-vec vs scalar-ref）：
 *   便宜档·大倍数是预期·【非硬赢】（[NG-4] kernel-axis SANITY·非 e2e·非 perf-covered）。
 *
 * ★ZERO-MODEL（[K-5] 正确性门）：scalar-ref 从【原始字节】b[]/a[]/c[] 独立重算整数结果，
 *   与向量核【不共享】任何 bit→lane 解码路径（向量核走 vand/vsrl/vwmul/vwredsum/vrgather）。
 *   纯整数无溢出 ⟹ out_ours[b] == out_scalar[b] 逐字节相等，任一 diff = 真发散。
 *
 * ★反空心（anti-hollow）：INJECT 注错证 oracle 有牙——
 *   INJECT=0 干净：expect mism=0（BYTE-EXACT OK）
 *   INJECT=1 腐蚀 OURS 输出：expect mism>0（证对拍咬得住 DUT 发散·GATE-OURS BITES）
 *   INJECT=2 腐蚀 ORACLE 常量：expect mism>0（证 oracle 非空心镜像·独立性 BITES）
 *
 * scalar-ref 标 noinline：与 extern 向量核对称（皆真函数调用），并使 objdump 探针可寻址。
 *
 * argv: <op:q4_0_nibble|offset_binary_n3|codebook_n3> <NB> <hot_iters> <cold_reps> <seed> [verify_only=0] [inject=0]
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define LANES 16
extern void weft_emitc_rvv_unsigned_nibble_q8_1_integer_core_kernel_rvv_unsigned_nibble_q8_1_integer_core(const uint8_t*,const int8_t*,const int8_t*,const int32_t*,int32_t*,size_t);
extern void weft_emitc_rvv_five_bit_q5_0_integer_core_kernel_rvv_five_bit_q5_0_integer_core(const uint8_t*,const int8_t*,const int8_t*,const int32_t*,int32_t*,size_t);
extern void weft_emitc_rvv_codebook_q8_0_integer_core_kernel_rvv_codebook_q8_0_integer_core(const uint8_t*,const int8_t*,const int8_t*,const int32_t*,int32_t*,size_t);

static const int8_t KV[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

/* ---- scalar-ref oracles (mirror the integer math exactly; noinline = symmetric call + probe-addressable) ---- */
static int g_oracle_fault = 0;   /* INJECT=2 poisons the oracle constant to prove independence */
__attribute__((noinline)) static void sc_nibble(const uint8_t*b,const int8_t*a,const int8_t*c,const int32_t*in,int32_t*out,size_t n){
    int32_t acc=in[0]; int adj = g_oracle_fault;
    for(size_t i=0;i<n;i++){ int lo=b[i]&0x0F, hi=(b[i]>>4)&0x0F; acc += (int)(int8_t)lo*(int)a[i] + (int)((int8_t)hi+adj)*(int)c[i]; }
    out[0]=acc;
}
__attribute__((noinline)) static void sc_offbin(const uint8_t*b,const int8_t*a,const int8_t*c,const int32_t*in,int32_t*out,size_t n){
    int32_t acc=in[0]; int adj = g_oracle_fault;
    uint32_t qhl=(uint16_t)*(const uint16_t*)(b+2), qhh=(uint16_t)*(const uint16_t*)(b+4);
    for(size_t i=0;i<n;i++){
        int lon=b[i]&0x0F, hin=(b[i]>>4)&0x0F;
        int lob=(qhl>>i)&1, hib=(qhh>>i)&1;
        int lo5=lon|(lob<<4), hi5=hin|(hib<<4);
        int lov=(int)(int8_t)lo5-16, hiv=(int)(int8_t)hi5-16+adj;
        acc += lov*(int)a[i] + hiv*(int)c[i];
    }
    out[0]=acc;
}
__attribute__((noinline)) static void sc_codebook(const uint8_t*b,const int8_t*a,const int8_t*c,const int32_t*in,int32_t*out,size_t n){
    int32_t acc=in[0]; int adj = g_oracle_fault;
    for(size_t i=0;i<n;i++){ int lo=b[i]&0x0F, hi=(b[i]>>4)&0x0F; acc += (int)KV[lo]*(int)a[i] + (int)(KV[hi]+adj)*(int)c[i]; }
    out[0]=acc;
}

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

enum { OP_NIB=0, OP_OFF, OP_CB, NOP };
static const char* ONAME[NOP]={"q4_0_nibble","offset_binary_n3","codebook_n3"};

int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: %s op NB hot cold seed [verify_only] [inject]\n",argv[0]); return 2; }
    int op=-1; for(int i=0;i<NOP;i++) if(!strcmp(argv[1],ONAME[i])) op=i;
    if(op<0){ fprintf(stderr,"bad op %s\n",argv[1]); return 2; }
    int NB=atoi(argv[2]),hiters=atoi(argv[3]),reps=atoi(argv[4]);
    rng=(uint64_t)strtoull(argv[5],0,0)|1ull;
    int vonly = argc>6?atoi(argv[6]):0;
    int inject = argc>7?atoi(argv[7]):0;
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    long vlen=(long)__riscv_vlenb()*8;

    size_t bB=(size_t)NB*LANES;
    uint8_t* B=aligned_alloc(64,bB);
    int8_t*  A=aligned_alloc(64,bB);
    int8_t*  C=aligned_alloc(64,bB);
    int32_t* Oo=aligned_alloc(64,(size_t)NB*4);
    int32_t* Os=aligned_alloc(64,(size_t)NB*4);
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!B||!A||!C||!Oo||!Os||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    for(size_t i=0;i<bB;i++){ B[i]=(uint8_t)xr(); A[i]=(int8_t)xr(); C[i]=(int8_t)xr(); }
    memset(g_flush,1,FLUSH_BYTES);
    const int32_t zero=0;

    #define OURS_BLOCK(bi) do{ const uint8_t*_b=B+(size_t)(bi)*LANES; const int8_t*_a=A+(size_t)(bi)*LANES,*_c=C+(size_t)(bi)*LANES; \
        switch(op){ \
          case OP_NIB: weft_emitc_rvv_unsigned_nibble_q8_1_integer_core_kernel_rvv_unsigned_nibble_q8_1_integer_core(_b,_a,_c,&zero,&Oo[bi],LANES); break; \
          case OP_OFF: weft_emitc_rvv_five_bit_q5_0_integer_core_kernel_rvv_five_bit_q5_0_integer_core(_b,_a,_c,&zero,&Oo[bi],LANES); break; \
          default:     weft_emitc_rvv_codebook_q8_0_integer_core_kernel_rvv_codebook_q8_0_integer_core(_b,_a,_c,&zero,&Oo[bi],LANES); } }while(0)
    #define SCAL_BLOCK(bi) do{ const uint8_t*_b=B+(size_t)(bi)*LANES; const int8_t*_a=A+(size_t)(bi)*LANES,*_c=C+(size_t)(bi)*LANES; \
        switch(op){ case OP_NIB: sc_nibble(_b,_a,_c,&zero,&Os[bi],LANES); break; \
          case OP_OFF: sc_offbin(_b,_a,_c,&zero,&Os[bi],LANES); break; \
          default: sc_codebook(_b,_a,_c,&zero,&Os[bi],LANES); } }while(0)

    /* INJECT=2 poisons the oracle (scalar-ref) constant BEFORE compute → oracle-fault BITES */
    g_oracle_fault = (inject==2) ? 1 : 0;

    for(int b=0;b<NB;b++){ OURS_BLOCK(b); SCAL_BLOCK(b); }

    /* INJECT=1 poisons OURS output AFTER compute → DUT-fault BITES */
    if(inject==1 && NB>0){ Oo[NB/2] += 1; }

    long long mism=0; int firsti=-1;
    for(int b=0;b<NB;b++){ if(Oo[b]!=Os[b]){ mism++; if(firsti<0)firsti=b; } }
    const char* verdict;
    if(inject==0) verdict = mism? "*** MISMATCH (UNEXPECTED) ***":"BYTE-EXACT-vs-scalar-ref OK";
    else          verdict = mism? "ANTI-HOLLOW BITES-OK (fault caught)":"*** HOLLOW: fault NOT caught ***";
    printf("VERIFY op=%s NB=%d LANES=%d inject=%d : int_mismatch=%lld first_b=%d sample_ours=%d scal=%d => %s\n",
           ONAME[op],NB,LANES,inject,mism,firsti,Oo[0],Os[0], verdict);
    if(vonly){ free(B);free(A);free(C);free(Oo);free(Os);free(g_flush); return (inject==0)?(mism?1:0):(mism?0:1); }

    double elems=(double)NB*LANES*2.0; /* products processed */
    double bytes=(double)bB*3.0 + (double)NB*4.0;
    (void)bytes;

    for(int w=0;w<3;w++){ for(int b=0;b<NB;b++) OURS_BLOCK(b); }
    double ours_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ for(int b=0;b<NB;b++) OURS_BLOCK(b);} double npc=(now_ns()-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; }
    for(int w=0;w<2;w++){ for(int b=0;b<NB;b++) SCAL_BLOCK(b); }
    double scal_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++){ for(int b=0;b<NB;b++) SCAL_BLOCK(b);} double npc=(now_ns()-t0)/(double)hiters; if(npc<scal_hot)scal_hot=npc; }

    double* ours=malloc(sizeof(double)*reps);
    double* scal=malloc(sizeof(double)*reps);
    cold_flush(); for(int b=0;b<NB;b++) OURS_BLOCK(b); cold_flush(); for(int b=0;b<NB;b++) SCAL_BLOCK(b);
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); for(int b=0;b<NB;b++) OURS_BLOCK(b); ours[p]=now_ns()-t0;
        cold_flush(); double t1=now_ns(); for(int b=0;b<NB;b++) SCAL_BLOCK(b); scal[p]=now_ns()-t1;
    }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(scal,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], sm=scal[reps/2];
    double o_q1=ours[reps/4], o_q3=ours[(3*reps)/4], s_q1=scal[reps/4], s_q3=scal[(3*reps)/4];
    double o_iqr = om>0?100.0*(o_q3-o_q1)/om:0.0, s_iqr = sm>0?100.0*(s_q3-s_q1)/sm:0.0;
    double ob=1e30,sb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(scal[p]<sb)sb=scal[p]; }

    /* ratio_cold = scalar-ref time / ours time = (elems/om)/(elems/sm) = sm/om ; >1 = ours faster */
    printf("PREDUCE op=%s VLEN=%ld NB=%d LANES=%d hiters=%d reps=%d seed=%s | "
           "HOT ours_ns=%.0f scal_ns=%.0f ratio_hot=%.4f | "
           "COLD ours_med_ns=%.0f o_iqr=%.2f scal_med_ns=%.0f s_iqr=%.2f ratio_cold_med=%.4f ratio_cold_best=%.4f | int_mismatch=%lld\n",
           ONAME[op],vlen,NB,LANES,hiters,reps,argv[5],
           ours_hot,scal_hot,(elems/ours_hot)/(elems/scal_hot),
           om,o_iqr,sm,s_iqr,(elems/om)/(elems/sm),(elems/ob)/(elems/sb),mism);
    free(B);free(A);free(C);free(Oo);free(Os);free(g_flush);free(ours);free(scal);
    return mism?1:0;
}
