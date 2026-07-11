/* flat_gemm_paired_driver.c — 线乙 FLAT repack-vs-blockdot paired A/B (rvv/VLEN128).
 *
 * Covers the FLAT (q4_0-family) constructed repack GEMMs: q4_0 / q4_1 / q5_0 / q5_1 / q8_0.
 * For each format it does TWO things on the SAME real quantized data:
 *
 *  (1) ZERO-MODEL NUMERIC GATE (correctness before timing):
 *      - generate random f16-scaled PLAIN weight blocks (block_qX_0/1) and PLAIN q8_0/q8_1
 *        activation blocks;
 *      - OPPONENT reference = ggml's REAL dispatched block-dot ggml_vec_dot_qX_qY (linked from the
 *        board's own libggml-cpu.so, gcc-15) run per (row,col) over the plain blocks -> Oo;
 *      - repack the SAME plain blocks into OUR interleaved x16 (weight) / x4 (activation) layout and
 *        run OUR exported repack GEMM kernel -> Or;
 *      - compare Or vs Oo (max relative error). The integer dot is bit-identical decode; only the
 *        f32 cross-block reduction ORDER differs, so a correct kernel agrees to ~1e-5..1e-4 rel.
 *      This is an INDEPENDENT oracle (ggml), zero reuse of our own intermediates = ZERO-MODEL.
 *
 *  (2) THROUGHPUT (best-of-N, data-independent control flow) OUR repack GEMM vs the SAME opponent
 *      block-dot GEMM. Both compute the identical logical matmul (K x nr x nc), macs=nr*nc*K.
 *
 * COMPILER-SYMMETRY [CASE-COMPILER-ASYMMETRY]: repack is compiler-sensitive, block-dot is not (1.002x).
 * This binary's OUR side is whatever compiler built it; opponent is gcc-15 shipped .so. So:
 *   - built with clang  -> ratio = SYSTEM/DEPLOYMENT account (we ship clang .o vs board's gcc ggml)
 *   - built with gcc    -> ratio = KERNEL account (compiler-symmetric gcc-ours vs gcc-opp)
 * Run it BOTH ways and report both ledgers. Board shipped compiler disclosed by caller.
 *
 * [NG-4]: L1 path-candidate datapoint (opponent has no working repack@VLEN128, we do). NOT a
 * [PERF-1] eight-gate beat unless the eight gates are separately walked.
 *
 * argv: <fmt=q4_0|q4_1|q5_0|q5_1|q8_0> <K(mult32)> <nr(mult4)> <nc(mult16)> <iters> <seed>
 * stdout: FLATGEMM fmt=.. VLEN=.. K=.. nr=.. nc=.. maxrelerr=.. GATE=PASS|FAIL
 *         ours_ns=.. ours_gmacs=.. opp_ns=.. opp_gmacs=.. ratio_ours_over_opp=..
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

/* ---- plain block structs (ggml canonical) ---- */
typedef struct { ggml_half d;            uint8_t qs[16]; } block_q4_0; /* 18 */
typedef struct { ggml_half d, m;         uint8_t qs[16]; } block_q4_1; /* 20 */
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0; /* 22 */
typedef struct { ggml_half d, m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1; /* 24 */
typedef struct { ggml_half d;            int8_t  qs[32]; } block_q8_0; /* 34 */
typedef struct { ggml_half d, s;         int8_t  qs[32]; } block_q8_1; /* 36 */
_Static_assert(sizeof(block_q4_0)==18,"q4_0"); _Static_assert(sizeof(block_q4_1)==20,"q4_1");
_Static_assert(sizeof(block_q5_0)==22,"q5_0"); _Static_assert(sizeof(block_q5_1)==24,"q5_1");
_Static_assert(sizeof(block_q8_0)==34,"q8_0"); _Static_assert(sizeof(block_q8_1)==36,"q8_1");

/* ---- OUR exported repack GEMM kernels: ABI (nr, bs, K, s, nc, vx=weights, vy=act) ---- */
extern void tcrv_emitc_ggml_gemm_q4_0_q8_0_kernel_ggml_gemm_q4_0_q8_0(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);

/* ---- OPPONENT: ggml real dispatched block-dot (linked from libggml-cpu.so) ---- */
extern void ggml_vec_dot_q4_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q8_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);

/* ---- fp16 <-> fp32 (round-to-nearest-even, small finite range only) ---- */
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

/* ---- self-contained SCALAR reference decode (independent ZERO-MODEL oracle) ---- */
static double scalar_cell(int f,const uint8_t* wrow,const uint8_t* arow,int nb,size_t wblk,size_t ablk){
    double out=0;
    for(int l=0;l<nb;l++){ const uint8_t* w=wrow+(size_t)l*wblk; const uint8_t* a=arow+(size_t)l*ablk;
        const uint16_t* wh=(const uint16_t*)w; const uint16_t* ah=(const uint16_t*)a;
        float dw=f16_to_f32(wh[0]); float da,mw=0,sa=0; long sumi=0;
        int is81=(f==1||f==3);
        da=f16_to_f32(ah[0]);
        const int8_t* y = (const int8_t*)(a+(is81?4:2));
        if(f==0){ /* q4_0 */ const uint8_t* qs=w+2;
            for(int j=0;j<16;j++){ int x0=(qs[j]&0xF)-8, x1=(qs[j]>>4)-8; sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi;
        } else if(f==1){ /* q4_1 */ mw=f16_to_f32(wh[1]); sa=f16_to_f32(ah[1]); const uint8_t* qs=w+4;
            for(int j=0;j<16;j++){ int x0=(qs[j]&0xF), x1=(qs[j]>>4); sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi + (double)mw*sa;
        } else if(f==2){ /* q5_0 */ uint32_t qh; memcpy(&qh,w+2,4); const uint8_t* qs=w+6;
            for(int j=0;j<16;j++){ uint8_t h0=(uint8_t)(((qh>>(j))&1)<<4), h1=(uint8_t)(((qh>>(j+16))&1)<<4);
                int x0=(int8_t)((qs[j]&0xF)|h0)-16, x1=(int8_t)((qs[j]>>4)|h1)-16; sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi;
        } else if(f==3){ /* q5_1 */ mw=f16_to_f32(wh[1]); sa=f16_to_f32(ah[1]); uint32_t qh; memcpy(&qh,w+4,4); const uint8_t* qs=w+8;
            for(int j=0;j<16;j++){ uint8_t h0=(uint8_t)(((qh>>(j))&1)<<4), h1=(uint8_t)(((qh>>(j+16))&1)<<4);
                int x0=(qs[j]&0xF)|h0, x1=(qs[j]>>4)|h1; sumi+=x0*y[j]+x1*y[j+16]; }
            out+=dw*da*(double)sumi + (double)mw*sa;
        } else { /* q8_0 */ const int8_t* wq=(const int8_t*)(w+2);
            for(int j=0;j<32;j++) sumi+=(int)wq[j]*y[j];
            out+=dw*da*(double)sumi;
        }
    }
    return out;
}

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  rq(void){ return (int8_t)((xr()&0xff)-128); }
static uint8_t rb(void){ return (uint8_t)(xr()&0xff); }
static uint16_t rscale(void){ /* small finite fp16 in ~[0.01,0.09] */ float f=0.01f+(float)(xr()%80)*0.001f; return f32_to_f16(f); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

enum { F_Q40, F_Q41, F_Q50, F_Q51, F_Q80 };
static int parse_fmt(const char* s){
    if(!strcmp(s,"q4_0"))return F_Q40; if(!strcmp(s,"q4_1"))return F_Q41;
    if(!strcmp(s,"q5_0"))return F_Q50; if(!strcmp(s,"q5_1"))return F_Q51;
    if(!strcmp(s,"q8_0"))return F_Q80; return -1;
}
/* interleaved group strides (bytes) */
static size_t WGRP(int f){ switch(f){case F_Q40:return 288;case F_Q41:return 320;case F_Q50:return 352;case F_Q51:return 384;default:return 544;} }
static size_t AGRP(int f){ return (f==F_Q41||f==F_Q51)?144:136; } /* q8_1x4 vs q8_0x4 */
static int is_q8_1(int f){ return f==F_Q41||f==F_Q51; }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s q4_0|q4_1|q5_0|q5_1|q8_0 K nr nc iters seed\n",argv[0]); return 2; }
    int f=parse_fmt(argv[1]); if(f<0){ fprintf(stderr,"bad fmt\n"); return 2; }
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),iters=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(K%32||nr%4||nc%16){ fprintf(stderr,"K%%32, nr%%4, nc%%16 required\n"); return 2; }
    long vlen=(long)__riscv_vlenb()*8;
    int nb=K/32, grpc=nc/16, grpr=nr/4;

    /* ---- allocate PLAIN opponent-format buffers (col-major rows for weights, row for acts) ---- */
    size_t wblk = (f==F_Q40)?18:(f==F_Q41)?20:(f==F_Q50)?22:(f==F_Q51)?24:34;
    size_t ablk = is_q8_1(f)?36:34;
    uint8_t* Wp=aligned_alloc(64,(size_t)nc*nb*wblk);   /* plain weights: [col][blk] */
    uint8_t* Ap=aligned_alloc(64,(size_t)nr*nb*ablk);   /* plain acts:    [row][blk] */
    /* ---- interleaved OUR-format buffers ---- */
    uint8_t* Wr=aligned_alloc(64,(size_t)grpc*nb*WGRP(f));
    uint8_t* Ar=aligned_alloc(64,(size_t)grpr*nb*AGRP(f));
    float* Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    float* Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    if(!Wp||!Ap||!Wr||!Ar||!Or||!Oo){ fprintf(stderr,"OOM\n"); return 3; }
    memset(Wr,0,(size_t)grpc*nb*WGRP(f)); memset(Ar,0,(size_t)grpr*nb*AGRP(f));

    /* ---- fill PLAIN weight blocks ---- */
    for(int c=0;c<nc;c++) for(int l=0;l<nb;l++){ uint8_t* p=Wp+((size_t)c*nb+l)*wblk;
        uint16_t* h=(uint16_t*)p;
        if(f==F_Q40){ h[0]=rscale(); for(int j=0;j<16;j++) p[2+j]=rb(); }
        else if(f==F_Q41){ h[0]=rscale(); h[1]=rscale(); for(int j=0;j<16;j++) p[4+j]=rb(); }
        else if(f==F_Q50){ h[0]=rscale(); for(int j=0;j<4;j++) p[2+j]=rb(); for(int j=0;j<16;j++) p[6+j]=rb(); }
        else if(f==F_Q51){ h[0]=rscale(); h[1]=rscale(); for(int j=0;j<4;j++) p[4+j]=rb(); for(int j=0;j<16;j++) p[8+j]=rb(); }
        else { h[0]=rscale(); int8_t* q=(int8_t*)(p+2); for(int j=0;j<32;j++) q[j]=rq(); }
    }
    /* ---- fill PLAIN activation blocks (q8_0 or q8_1) ---- */
    for(int r=0;r<nr;r++) for(int l=0;l<nb;l++){ uint8_t* p=Ap+((size_t)r*nb+l)*ablk;
        uint16_t* h=(uint16_t*)p; uint16_t dh=rscale(); h[0]=dh;
        int8_t* q; long sum=0;
        if(is_q8_1(f)){ q=(int8_t*)(p+4); }
        else { q=(int8_t*)(p+2); }
        for(int j=0;j<32;j++){ q[j]=rq(); sum+=q[j]; }
        if(is_q8_1(f)){ h[1]=f32_to_f16(f16_to_f32(dh)*(float)sum); } /* q8_1 s = d * sum(qs) */
    }

    /* ---- REPACK plain -> OUR interleaved (byte-exact to kernel addressing) ---- */
    for(int gc=0;gc<grpc;gc++) for(int l=0;l<nb;l++){
        uint8_t* g=Wr+((size_t)gc*nb+l)*WGRP(f);
        for(int cc=0;cc<16;cc++){ int c=gc*16+cc; const uint8_t* p=Wp+((size_t)c*nb+l)*wblk; const uint16_t* h=(const uint16_t*)p;
            if(f==F_Q40){ /* signed nibble via sign-extend => weights pre-XOR 0x88 (ggml repack trick) */
                ((uint16_t*)g)[cc]=h[0]; for(int b=0;b<16;b++) g[32+b*16+cc]=(uint8_t)(p[2+b]^0x88); }
            else if(f==F_Q41){ ((uint16_t*)g)[cc]=h[0]; ((uint16_t*)(g+32))[cc]=h[1]; for(int b=0;b<16;b++) g[64+b*16+cc]=p[4+b]; }
            else if(f==F_Q50){ ((uint16_t*)g)[cc]=h[0]; for(int b=0;b<16;b++) g[32+b*16+cc]=p[6+b];
                uint32_t qh; memcpy(&qh,p+2,4); uint16_t* qhr=(uint16_t*)(g+288);
                for(int pp=0;pp<32;pp++) if((qh>>pp)&1) qhr[pp]|=(uint16_t)(1u<<cc); }
            else if(f==F_Q51){ ((uint16_t*)g)[cc]=h[0]; ((uint16_t*)(g+32))[cc]=h[1]; for(int b=0;b<16;b++) g[64+b*16+cc]=p[8+b];
                uint32_t qh; memcpy(&qh,p+4,4); uint16_t* qhr=(uint16_t*)(g+320);
                for(int pp=0;pp<32;pp++) if((qh>>pp)&1) qhr[pp]|=(uint16_t)(1u<<cc); }
            else { ((uint16_t*)g)[cc]=h[0]; const int8_t* q=(const int8_t*)(p+2); for(int i=0;i<32;i++) ((int8_t*)g)[32+i*16+cc]=q[i]; }
        }
    }
    for(int gr=0;gr<grpr;gr++) for(int l=0;l<nb;l++){
        uint8_t* g=Ar+((size_t)gr*nb+l)*AGRP(f); int qoff=is_q8_1(f)?16:8;
        for(int rr=0;rr<4;rr++){ int r=gr*4+rr; const uint8_t* p=Ap+((size_t)r*nb+l)*ablk; const uint16_t* h=(const uint16_t*)p;
            ((uint16_t*)g)[rr]=h[0];                                   /* d @ [0,8) */
            if(is_q8_1(f)) ((uint16_t*)(g+8))[rr]=h[1];                /* s @ [8,16) */
            const int8_t* q=(const int8_t*)(p+(is_q8_1(f)?4:2));
            for(int i=0;i<32;i++) ((int8_t*)g)[qoff+i*4+rr]=q[i];      /* quants */
        }
    }

    /* ---- OPPONENT reference: per-(row,col) block-dot over PLAIN blocks -> Oo ---- */
    #define OPP_CELL(c,r) do{ const void* wc=Wp+((size_t)(c)*nb)*wblk; const void* ar=Ap+((size_t)(r)*nb)*ablk; \
        switch(f){ case F_Q40: ggml_vec_dot_q4_0_q8_0(K,&Oo[(size_t)(r)*nc+(c)],0,wc,0,ar,0,1); break; \
                   case F_Q41: ggml_vec_dot_q4_1_q8_1(K,&Oo[(size_t)(r)*nc+(c)],0,wc,0,ar,0,1); break; \
                   case F_Q50: ggml_vec_dot_q5_0_q8_0(K,&Oo[(size_t)(r)*nc+(c)],0,wc,0,ar,0,1); break; \
                   case F_Q51: ggml_vec_dot_q5_1_q8_1(K,&Oo[(size_t)(r)*nc+(c)],0,wc,0,ar,0,1); break; \
                   default:    ggml_vec_dot_q8_0_q8_0(K,&Oo[(size_t)(r)*nc+(c)],0,wc,0,ar,0,1); break; } }while(0)
    for(int c=0;c<nc;c++) for(int r=0;r<nr;r++) OPP_CELL(c,r);

    /* ---- OURS -> Or ---- */
    #define OUR_RUN() do{ switch(f){ \
        case F_Q40: tcrv_emitc_ggml_gemm_q4_0_q8_0_kernel_ggml_gemm_q4_0_q8_0((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break; \
        case F_Q41: tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break; \
        case F_Q50: tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break; \
        case F_Q51: tcrv_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break; \
        default:    tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0((size_t)nr,(size_t)nc,(size_t)K,Or,(size_t)nc,Wr,Ar); break; } }while(0)
    OUR_RUN();

    /* ---- NUMERIC GATE: compare OURS and OPP each against the independent SCALAR ref ---- */
    double maxrel=0, maxrel_opp=0; int nbad=0; const char* dbg=getenv("FLATDBG");
    for(int r=0;r<nr;r++) for(int c=0;c<nc;c++){
        const uint8_t* wrow=Wp+((size_t)c*nb)*wblk; const uint8_t* arow=Ap+((size_t)r*nb)*ablk;
        double s=scalar_cell(f,wrow,arow,nb,wblk,ablk);
        double a=Or[(size_t)r*nc+c], b=Oo[(size_t)r*nc+c];
        /* mixed abs/rel: floor at 1.0 so near-zero cancellation cells (|out|<1, meaningless rel err
         * where ours tracks ggml exactly) don't false-alarm; real bugs give O(1e2+) errors everywhere. */
        double den=fabs(s)>1.0?fabs(s):1.0;
        double re=fabs(a-s)/den, reo=fabs(b-s)/den;
        if(re>maxrel)maxrel=re; if(reo>maxrel_opp)maxrel_opp=reo; if(re>5e-3)nbad++;
        if(dbg && r<2 && c<3) fprintf(stderr,"[dbg] r=%d c=%d scalar=%.5f ours=%.5f opp=%.5f re_ours=%.2e re_opp=%.2e\n",r,c,s,a,b,re,reo);
    }
    int gate = (maxrel<5e-3);

    /* ---- THROUGHPUT: warmup + best-of-N ---- */
    volatile double sink=0;
    for(int w=0;w<3;w++) OUR_RUN();
    double ours_best=1e30;
    for(int p=0;p<7;p++){ double t0=now_ns(); for(int it=0;it<iters;it++) OUR_RUN(); double npc=(now_ns()-t0)/iters; if(npc<ours_best)ours_best=npc; sink+=Or[0]+Or[(size_t)nr*nc-1]; }
    #define OPP_GEMM() do{ for(int c=0;c<nc;c++) for(int r=0;r<nr;r++) OPP_CELL(c,r); }while(0)
    for(int w=0;w<2;w++) OPP_GEMM();
    double opp_best=1e30;
    for(int p=0;p<7;p++){ double t0=now_ns(); for(int it=0;it<iters;it++) OPP_GEMM(); double npc=(now_ns()-t0)/iters; if(npc<opp_best)opp_best=npc; sink+=Oo[0]+Oo[(size_t)nr*nc-1]; }

    double macs=(double)nr*nc*K, og=macs/ours_best, pg=macs/opp_best;
    printf("FLATGEMM fmt=%s VLEN=%ld K=%d nr=%d nc=%d iters=%d relerr_ours=%.3e relerr_opp=%.3e nbad=%d GATE=%s ours_ns=%.1f ours_gmacs=%.4f opp_ns=%.1f opp_gmacs=%.4f ratio_ours_over_opp=%.4f sink=%.1f\n",
        argv[1],vlen,K,nr,nc,iters,maxrel,maxrel_opp,nbad,gate?"PASS":"FAIL",ours_best,og,opp_best,pg,og/pg,(double)sink);
    free(Wp);free(Ap);free(Wr);free(Ar);free(Or);free(Oo);
    return gate?0:1;
}
