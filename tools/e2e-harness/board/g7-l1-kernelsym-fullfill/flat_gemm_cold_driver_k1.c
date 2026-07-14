/* flat_gemm_cold_driver_k1.c — G7 L1 货架A kernel-sym FLAT q4_1/q5_0/q5_1 hot/cold micro A/B
 * @k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / clang-18 symmetric.
 *
 * PORT of rvv-batch/flat_gemm_cold_driver.c (VLEN128 vl=8) → k1 VLEN256 vl=16 kernel-axis.
 * ★KEY: interleaved weight layout `block_qX_1x16` (16-col group) is BYTE-IDENTICAL between
 * VLEN128 (kernel reads two 8-lane halves) and VLEN256 (kernel reads one 16-lane strip) —
 * only the emitted kernel's strip-read-width differs (vl=8 → vl=16), NOT the memory layout.
 * So the same repack_w() feeds the k1 vl=16 emitted kernel; the ZERO-MODEL numeric gate
 * (independent fp64 scalar decoder vs ours vs opponent) is the byte-exact safety net.
 *
 * COLD protocol: POOL of P weight tiles, footprint >> k1 L2 (512KiB, NO L3) so every round
 * streams each weight tile cold from DRAM. Activations single-copy (warm-act/cold-weight = decode).
 * Same-shape (VERIFY-LADDER G2): GEMM form, nr CLI arg => nr-shape {4,16,64} variant sweep in one binary.
 *
 * OURS  = weft VLEN256 vl=16 emitted repack GEMM (weft_emitted_gemm_qX.inc, clang-18, march-symmetric).
 * OPP   = k1 stock dispatched block-dot ggml_vec_dot_qX_qY (libggml-cpu.so, clang-18). k1 ships ZERO
 *         q4_1/q5_0/q5_1 repack => single-implementation compromise (折中态) => our repack = only repack.
 * [CASE-COMPILER-ASYMMETRY] not triggered: ours + opp both clang-18 (k1 shipped compiler).
 * [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win, NOT perf-covered.
 *
 * argv: <fmt=q4_1|q5_0|q5_1> <K(mult32)> <nr(mult4)> <nc(mult16)> <hot_iters> <pool_tiles> <rounds> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
typedef struct { ggml_half d, m;         uint8_t qs[16]; } block_q4_1; /* 20 */
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0; /* 22 */
typedef struct { ggml_half d, m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1; /* 24 */
typedef struct { ggml_half d, s;         int8_t  qs[32]; } block_q8_1; /* 36 */
_Static_assert(sizeof(block_q4_1)==20,"q4_1");
_Static_assert(sizeof(block_q5_0)==22,"q5_0"); _Static_assert(sizeof(block_q5_1)==24,"q5_1");
_Static_assert(sizeof(block_q8_1)==36,"q8_1");

/* OUR exported repack GEMM kernels (weft VLEN256 vl=16; kernel body from weft_emitted_gemm_qX.inc). */
extern void weft_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);

/* OPPONENT: k1 stock block-dot (machine-probed public T symbols) */
extern void ggml_vec_dot_q4_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint16_t f32_to_f16(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; man|=0x800000u; uint32_t sh=(uint32_t)(14-exp);
        uint16_t r=(uint16_t)(man>>sh); if((man>>(sh-1))&1) r++; return (uint16_t)(sign|r); }
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    uint16_t r=(uint16_t)(sign|((uint32_t)exp<<10)|(man>>13)); if((man>>12)&1) r++; return r;
}
static float f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16; uint32_t exp=(h>>10)&0x1f; uint32_t man=h&0x3ff; uint32_t o;
    if(exp==0){ if(man==0){o=sign;} else { exp=127-15+1; while(!(man&0x400)){man<<=1;exp--;} man&=0x3ff; o=sign|(exp<<23)|(man<<13);} }
    else if(exp==31){ o=sign|0x7f800000u|(man<<13); }
    else { o=sign|((exp+112)<<23)|(man<<13); }
    float f; memcpy(&f,&o,4); return f;
}
/* independent fp64 scalar decoder (ZERO-MODEL): f=1 q4_1, f=2 q5_0, f=3 q5_1 */
static double scalar_cell(int f,const uint8_t* wrow,const uint8_t* arow,int nb,size_t wblk,size_t ablk){
    double out=0;
    for(int l=0;l<nb;l++){ const uint8_t* w=wrow+(size_t)l*wblk; const uint8_t* a=arow+(size_t)l*ablk;
        const uint16_t* wh=(const uint16_t*)w; const uint16_t* ah=(const uint16_t*)a;
        float dw=f16_to_f32(wh[0]); float da,mw=0,sa=0; long sumi=0;
        int is81=(f==1||f==3);
        da=f16_to_f32(ah[0]);
        const int8_t* y = (const int8_t*)(a+(is81?4:2));
        if(f==1){ mw=f16_to_f32(wh[1]); sa=f16_to_f32(ah[1]); const uint8_t* qs=w+4;
            for(int j=0;j<16;j++){ int x0=(qs[j]&0xF), x1=(qs[j]>>4); sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi + (double)mw*sa;
        } else if(f==2){ uint32_t qh; memcpy(&qh,w+2,4); const uint8_t* qs=w+6;
            for(int j=0;j<16;j++){ uint8_t h0=(uint8_t)(((qh>>(j))&1)<<4), h1=(uint8_t)(((qh>>(j+16))&1)<<4);
                int x0=(int8_t)((qs[j]&0xF)|h0)-16, x1=(int8_t)((qs[j]>>4)|h1)-16; sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi;
        } else { /* f==3 q5_1 */ mw=f16_to_f32(wh[1]); sa=f16_to_f32(ah[1]); uint32_t qh; memcpy(&qh,w+4,4); const uint8_t* qs=w+8;
            for(int j=0;j<16;j++){ uint8_t h0=(uint8_t)(((qh>>(j))&1)<<4), h1=(uint8_t)(((qh>>(j+16))&1)<<4);
                int x0=(qs[j]&0xF)|h0, x1=(qs[j]>>4)|h1; sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi + (double)mw*sa;
        }
    }
    return out;
}
static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  rq(void){ return (int8_t)((xr()&0xff)-128); }
static uint8_t rb(void){ return (uint8_t)(xr()&0xff); }
static uint16_t rscale(void){ float f=0.01f+(float)(xr()%80)*0.001f; return f32_to_f16(f); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){ qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]); double q1=v[n/4],q3=v[(3*n)/4]; *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0; }

enum { F_Q41=1, F_Q50=2, F_Q51=3 };
static int parse_fmt(const char* s){
    if(!strcmp(s,"q4_1"))return F_Q41;
    if(!strcmp(s,"q5_0"))return F_Q50; if(!strcmp(s,"q5_1"))return F_Q51;
    return -1;
}
static size_t WGRP(int f){ switch(f){case F_Q41:return 320;case F_Q50:return 352;default:return 384;} }
static size_t AGRP(int f){ return (f==F_Q41||f==F_Q51)?144:136; }
static int is_q8_1(int f){ return f==F_Q41||f==F_Q51; }
static size_t WBLK(int f){ return (f==F_Q41)?20:(f==F_Q50)?22:24; }

/* repack one plain weight tile Wp -> interleaved Wr (byte-exact to kernel addressing; block_qX_1x16) */
static void repack_w(int f,const uint8_t* Wp,uint8_t* Wr,int nc,int nb){
    int grpc=nc/16;
    memset(Wr,0,(size_t)grpc*nb*WGRP(f));
    size_t wblk=WBLK(f);
    for(int gc=0;gc<grpc;gc++) for(int l=0;l<nb;l++){
        uint8_t* g=Wr+((size_t)gc*nb+l)*WGRP(f);
        for(int cc=0;cc<16;cc++){ int c=gc*16+cc; const uint8_t* p=Wp+((size_t)c*nb+l)*wblk; const uint16_t* h=(const uint16_t*)p;
            if(f==F_Q41){ ((uint16_t*)g)[cc]=h[0]; ((uint16_t*)(g+32))[cc]=h[1]; for(int b=0;b<16;b++) g[64+b*16+cc]=p[4+b]; }
            else if(f==F_Q50){ ((uint16_t*)g)[cc]=h[0]; for(int b=0;b<16;b++) g[32+b*16+cc]=p[6+b];
                uint32_t qh; memcpy(&qh,p+2,4); uint16_t* qhr=(uint16_t*)(g+288);
                for(int pp=0;pp<32;pp++) if((qh>>pp)&1) qhr[pp]|=(uint16_t)(1u<<cc); }
            else { /* F_Q51 */ ((uint16_t*)g)[cc]=h[0]; ((uint16_t*)(g+32))[cc]=h[1]; for(int b=0;b<16;b++) g[64+b*16+cc]=p[8+b];
                uint32_t qh; memcpy(&qh,p+4,4); uint16_t* qhr=(uint16_t*)(g+320);
                for(int pp=0;pp<32;pp++) if((qh>>pp)&1) qhr[pp]|=(uint16_t)(1u<<cc); }
        }
    }
}
static void our_run(int f,int nr,int nc,int K,float* Or,const uint8_t* Wr,const uint8_t* Ar){
    switch(f){
      case F_Q41: weft_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break;
      case F_Q50: weft_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break;
      default:    weft_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break;
    }
}
static void opp_run(int f,int nr,int nc,int K,float* Oo,const uint8_t* Wp,const uint8_t* Ap,int nb){
    size_t wblk=WBLK(f), ablk=is_q8_1(f)?36:34;
    for(int c=0;c<nc;c++) for(int r=0;r<nr;r++){
        const void* wc=Wp+((size_t)c*nb)*wblk; const void* ar=Ap+((size_t)r*nb)*ablk;
        switch(f){ case F_Q41: ggml_vec_dot_q4_1_q8_1(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); break;
                   case F_Q50: ggml_vec_dot_q5_0_q8_0(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); break;
                   default:    ggml_vec_dot_q5_1_q8_1(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); break; }
    }
}

int main(int argc,char**argv){
    if(argc<9){ fprintf(stderr,"usage: %s fmt K nr nc hot_iters pool_tiles rounds seed\n",argv[0]); return 2; }
    int f=parse_fmt(argv[1]); if(f<0){ fprintf(stderr,"bad fmt (q4_1|q5_0|q5_1)\n"); return 2; }
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),hiters=atoi(argv[5]),P=atoi(argv[6]),rounds=atoi(argv[7]);
    rng=(uint64_t)strtoull(argv[8],0,0)|1ull;
    if(K%32||nr%4||nc%16){ fprintf(stderr,"K%%32,nr%%4,nc%%16 required\n"); return 2; }
    if(P<1)P=1; if(rounds<8)rounds=8; if(hiters<1)hiters=1;
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/32, grpc=nc/16, grpr=nr/4;
    size_t wblk=WBLK(f), ablk=is_q8_1(f)?36:34;
    size_t wr_bytes=(size_t)grpc*nb*WGRP(f);
    size_t wp_bytes=(size_t)nc*nb*wblk;      /* plain weight footprint per tile (roofline denominator) */

    /* single activation (warm) */
    uint8_t* Ap=aligned_alloc(64,(size_t)nr*nb*ablk);
    uint8_t* Ar=aligned_alloc(64,(size_t)grpr*nb*AGRP(f));
    float* Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    float* Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    uint8_t** Wp=malloc(P*sizeof(uint8_t*));
    uint8_t** Wr=malloc(P*sizeof(uint8_t*));
    if(!Ap||!Ar||!Or||!Oo||!Wp||!Wr){ fprintf(stderr,"OOM meta\n"); return 3; }
    memset(Ar,0,(size_t)grpr*nb*AGRP(f));

    for(int r=0;r<nr;r++) for(int l=0;l<nb;l++){ uint8_t* p=Ap+((size_t)r*nb+l)*ablk;
        uint16_t* h=(uint16_t*)p; uint16_t dh=rscale(); h[0]=dh; int8_t* q; long sum=0;
        q=(int8_t*)(p+(is_q8_1(f)?4:2)); for(int j=0;j<32;j++){ q[j]=rq(); sum+=q[j]; }
        if(is_q8_1(f)){ h[1]=f32_to_f16(f16_to_f32(dh)*(float)sum); } }
    for(int gr=0;gr<grpr;gr++) for(int l=0;l<nb;l++){ uint8_t* g=Ar+((size_t)gr*nb+l)*AGRP(f); int qoff=is_q8_1(f)?16:8;
        for(int rr=0;rr<4;rr++){ int r=gr*4+rr; const uint8_t* p=Ap+((size_t)r*nb+l)*ablk; const uint16_t* h=(const uint16_t*)p;
            ((uint16_t*)g)[rr]=h[0]; if(is_q8_1(f)) ((uint16_t*)(g+8))[rr]=h[1];
            const int8_t* q=(const int8_t*)(p+(is_q8_1(f)?4:2)); for(int i=0;i<32;i++) ((int8_t*)g)[qoff+i*4+rr]=q[i]; } }

    for(int p=0;p<P;p++){
        Wp[p]=aligned_alloc(64,wp_bytes); Wr[p]=aligned_alloc(64,wr_bytes);
        if(!Wp[p]||!Wr[p]){ fprintf(stderr,"OOM pool %d (need ~%.0f MB)\n",p,(double)P*(wp_bytes+wr_bytes)/1e6); return 3; }
        for(int c=0;c<nc;c++) for(int l=0;l<nb;l++){ uint8_t* q=Wp[p]+((size_t)c*nb+l)*wblk; uint16_t* h=(uint16_t*)q;
            if(f==F_Q41){ h[0]=rscale(); h[1]=rscale(); for(int j=0;j<16;j++) q[4+j]=rb(); }
            else if(f==F_Q50){ h[0]=rscale(); for(int j=0;j<4;j++) q[2+j]=rb(); for(int j=0;j<16;j++) q[6+j]=rb(); }
            else { h[0]=rscale(); h[1]=rscale(); for(int j=0;j<4;j++) q[4+j]=rb(); for(int j=0;j<16;j++) q[8+j]=rb(); } }
        repack_w(f,Wp[p],Wr[p],nc,nb);
    }

    /* ---- ZERO-MODEL numeric gate on tile 0 ---- */
    our_run(f,nr,nc,K,Or,Wr[0],Ar);
    opp_run(f,nr,nc,K,Oo,Wp[0],Ap,nb);
    double maxrel=0,maxrel_opp=0; int nbad=0;
    for(int r=0;r<nr;r++) for(int c=0;c<nc;c++){
        const uint8_t* wrow=Wp[0]+((size_t)c*nb)*wblk; const uint8_t* arow=Ap+((size_t)r*nb)*ablk;
        double s=scalar_cell(f,wrow,arow,nb,wblk,ablk);
        double a=Or[(size_t)r*nc+c], b=Oo[(size_t)r*nc+c];
        double den=fabs(s)>1.0?fabs(s):1.0;
        double re=fabs(a-s)/den, reo=fabs(b-s)/den;
        if(re>maxrel)maxrel=re; if(reo>maxrel_opp)maxrel_opp=reo; if(re>5e-3)nbad++;
    }
    int gate=(maxrel<5e-3);
    double macs=(double)nr*nc*K;

    /* ---- HOT: single tile (tile 0) best-of-hiters ---- */
    volatile double sink=0;
    for(int w=0;w<3;w++){ our_run(f,nr,nc,K,Or,Wr[0],Ar); opp_run(f,nr,nc,K,Oo,Wp[0],Ap,nb); }
    double ours_hot=1e30, opp_hot=1e30;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns(); for(int it=0;it<hiters;it++) our_run(f,nr,nc,K,Or,Wr[0],Ar); double t1=now_ns();
        double npc=(t1-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; sink+=Or[0];
        double t2=now_ns(); for(int it=0;it<hiters;it++) opp_run(f,nr,nc,K,Oo,Wp[0],Ap,nb); double t3=now_ns();
        npc=(t3-t2)/(double)hiters; if(npc<opp_hot)opp_hot=npc; sink+=Oo[0];
    }
    printf("HOT fmt=%s VLEN=%ld K=%d nr=%d nc=%d relerr_ours=%.3e relerr_opp=%.3e nbad=%d GATE=%s "
           "ours_ns=%.1f ours_gmacs=%.4f opp_ns=%.1f opp_gmacs=%.4f ratio_ours_over_opp=%.4f sink=%.1f\n",
           argv[1],vlen,K,nr,nc,maxrel,maxrel_opp,nbad,gate?"PASS":"FAIL",
           ours_hot,macs/ours_hot,opp_hot,macs/opp_hot,(macs/ours_hot)/(macs/opp_hot),(double)sink);

    /* ---- COLD: pool median-of-rounds (weights streamed cold from DRAM; k1 L2=512KiB no L3) ---- */
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++) our_run(f,nr,nc,K,Or,Wr[p],Ar); for(int p=0;p<P;p++) opp_run(f,nr,nc,K,Oo,Wp[p],Ap,nb); sink+=Or[0]+Oo[0]; }
    double *to=malloc(rounds*sizeof(double)), *tp=malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++) our_run(f,nr,nc,K,Or,Wr[p],Ar); double t1=now_ns(); to[rd]=(t1-t0)/(double)P; sink+=Or[0];
        double t2=now_ns(); for(int p=0;p<P;p++) opp_run(f,nr,nc,K,Oo,Wp[p],Ap,nb); double t3=now_ns(); tp[rd]=(t3-t2)/(double)P; sink+=Oo[0];
    }
    double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
    printf("COLD fmt=%s VLEN=%ld K=%d nr=%d nc=%d pool=%d ws_ours_MB=%.1f ws_opp_MB=%.1f wp_bytes=%zu "
           "ours_ns_med=%.1f ours_iqrpct=%.2f ours_gmacs=%.4f ours_wGBs=%.3f "
           "opp_ns_med=%.1f opp_iqrpct=%.2f opp_gmacs=%.4f opp_wGBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           argv[1],vlen,K,nr,nc,P,(double)P*wr_bytes/1e6,(double)P*wp_bytes/1e6,wp_bytes,
           om,oi,macs/om,(double)wp_bytes/om, pm,pi,macs/pm,(double)wp_bytes/pm,(macs/om)/(macs/pm),(double)sink);
    return gate?0:1;
}
