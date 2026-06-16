// inc13-gemv-research: Q4_0 x Q8_0 GEMM / PREFILL weight-decode-reuse probe.
//
// THROWAWAY probe. The GEMV probes showed activation reuse is worthless (compute-bound).
// The MIRROR question: in PREFILL (one weight matrix x M activation columns), can we
// DECODE each weight block ONCE and reuse the decoded lanes across M activation columns,
// vs ggml's path which re-decodes the weight for EVERY column (M x vec_dot)?
//
// Compute-bound cuts the other way here: weight decode (vand/vsrl/vsub ~ half the integer
// vector work) is REUSED M-fold instead of zero-fold. That is the GEMM lever.
//
// LAYOUT: standard ggml q4_0 weights (AoS), standard q8_0 activations (AoS), M activation
// columns sharing one weight row. NO interleave, NO repack, NO reduction tree -- reuses
// exactly the per-row vec_dot machinery, just hoists the weight decode out of the M loop.
//
// VARIANTS (produce M outputs for ONE weight row x M activation columns):
//   a_Mxvecdot : ggml per-row vec_dot called M times (re-decodes weight every column).
//                THE PREFILL BASELINE (this is what runs today at VLEN=128, since the
//                repack gemm is disabled -- gemm also falls back to per-(row,col) vec_dot).
//   w_hoist    : decode each weight block ONCE, inner loop over M activation columns does
//                vwmul/vwmacc/vwredsum + per-col scalar fold. Weight decode amortized M-fold.
//
// BYTE-EXACT REF = a_Mxvecdot. Each (row,col) folds blocks ascending: sumi*d_wt*d_act.
// w_hoist keeps that exact order -> bitwise equal.
//
// Board: ssh rvv, VLEN=128. Build: clang -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast
// Run pinned: taskset -c 3 ./gemm_probe <K> <M_cols> <iters> <repeats>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 18
#define Q8_STRIDE 34

static inline float fp16_to_fp32(uint16_t h){ _Float16 hf; memcpy(&hf,&h,2); return (float)hf; }
static inline uint16_t fp32_to_fp16(float f){ _Float16 hf=(_Float16)f; uint16_t h; memcpy(&h,&hf,2); return h; }

// ggml per-row vec_dot (one weight row x one activation col)
static void vec_dot(int n, float *s, const uint8_t *x, const uint8_t *y){
  const int nb=n/QK; const size_t vl=QK/2; float sumf=0.0f;
  for(int ib=0;ib<nb;++ib){
    const uint8_t *xb=x+(size_t)ib*Q4_STRIDE, *yb=y+(size_t)ib*Q8_STRIDE;
    vuint8m1_t tx=__riscv_vle8_v_u8m1(xb+2,vl);
    vint8m1_t y0=__riscv_vle8_v_i8m1((const int8_t*)(yb+2),vl);
    vint8m1_t y1=__riscv_vle8_v_i8m1((const int8_t*)(yb+2+16),vl);
    vuint8m1_t xa=__riscv_vand_vx_u8m1(tx,0x0F,vl), xl=__riscv_vsrl_vx_u8m1(tx,4,vl);
    vint8m1_t v0=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xa),8,vl);
    vint8m1_t v1=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xl),8,vl);
    vint16m2_t m=__riscv_vwmul_vv_i16m2(v0,y0,vl);
    m=__riscv_vwmacc_vv_i16m2(m,v1,y1,vl);
    vint32m1_t z=__riscv_vmv_v_x_i32m1(0,vl);
    int sumi=__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(m,z,vl));
    uint16_t dx,dy; memcpy(&dx,xb,2); memcpy(&dy,yb,2);
    sumf += sumi*fp16_to_fp32(dx)*fp16_to_fp32(dy);
  }
  *s=sumf;
}

// ============ a_Mxvecdot: M x vec_dot (re-decodes weight every column) ============
// one weight row W, M activation columns Y[0..M-1] (each its own q8_0 stream) -> M outputs.
static void gemm_a(int K, int M, float *s, const uint8_t *W, const uint8_t *const *Y){
  for(int c=0;c<M;++c) vec_dot(K, &s[c], W, Y[c]);
}

// ============ w_hoist: decode weight block ONCE, reuse across M columns ============
static void gemm_w_hoist(int K, int M, float *s, const uint8_t *W, const uint8_t *const *Y){
  const int nb=K/QK; const size_t vl=QK/2;
  for(int c=0;c<M;++c) s[c]=0.0f;
  for(int ib=0;ib<nb;++ib){
    const uint8_t *xb=W+(size_t)ib*Q4_STRIDE;
    // decode the weight block ONCE: v0/v1 are reused for all M columns
    vuint8m1_t tx=__riscv_vle8_v_u8m1(xb+2,vl);
    vuint8m1_t xa=__riscv_vand_vx_u8m1(tx,0x0F,vl), xl=__riscv_vsrl_vx_u8m1(tx,4,vl);
    vint8m1_t v0=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xa),8,vl);
    vint8m1_t v1=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xl),8,vl);
    uint16_t dxb; memcpy(&dxb,xb,2); float dx=fp16_to_fp32(dxb);
    for(int c=0;c<M;++c){
      const uint8_t *yb=Y[c]+(size_t)ib*Q8_STRIDE;
      vint8m1_t y0=__riscv_vle8_v_i8m1((const int8_t*)(yb+2),vl);
      vint8m1_t y1=__riscv_vle8_v_i8m1((const int8_t*)(yb+2+16),vl);
      vint16m2_t m=__riscv_vwmul_vv_i16m2(v0,y0,vl);   // reuse decoded v0/v1
      m=__riscv_vwmacc_vv_i16m2(m,v1,y1,vl);
      vint32m1_t z=__riscv_vmv_v_x_i32m1(0,vl);
      int sumi=__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(m,z,vl));
      uint16_t dyb; memcpy(&dyb,yb,2);
      s[c] += sumi*dx*fp16_to_fp32(dyb);               // sumi*d_wt*d_act (vec_dot order)
    }
  }
}

// ============================ harness ============================
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static void fill_q4(uint8_t *row,int nb,unsigned *sd){
  for(int b=0;b<nb;++b){ uint8_t *k=row+(size_t)b*Q4_STRIDE;
    uint16_t d=fp32_to_fp16(0.005f+(rand_r(sd)%100)*0.0003f); memcpy(k,&d,2);
    for(int j=0;j<QK/2;++j) k[2+j]=(uint8_t)(rand_r(sd)&0xFF);} }
static void fill_q8(uint8_t *y,int nb,unsigned *sd){
  for(int b=0;b<nb;++b){ uint8_t *k=y+(size_t)b*Q8_STRIDE;
    uint16_t d=fp32_to_fp16(0.01f+(rand_r(sd)%100)*0.0004f); memcpy(k,&d,2);
    for(int j=0;j<QK;++j) ((int8_t*)(k+2))[j]=(int8_t)((rand_r(sd)%255)-127);} }

typedef void (*gfn)(int,int,float*,const uint8_t*,const uint8_t*const*);
static double bench(gfn fn,int K,int M,float*s,const uint8_t*W,const uint8_t*const*Y,long it,int rp){
  double best=1e30;
  for(int r=0;r<rp;++r){ double t0=now_ns(); for(long i=0;i<it;++i) fn(K,M,s,W,Y);
    double t1=now_ns(); double per=(t1-t0)/it; if(per<best)best=per; }
  return best;
}

int main(int argc,char**argv){
  int K=argc>1?atoi(argv[1]):4096;
  int M=argc>2?atoi(argv[2]):8;
  long iters=argc>3?atol(argv[3]):4000;
  int reps=argc>4?atoi(argv[4]):20;
  if(K%QK){fprintf(stderr,"K%%32\n");return 1;}
  int nb=K/QK;
  uint8_t *W=malloc((size_t)nb*Q4_STRIDE);
  uint8_t **Y=malloc((size_t)M*sizeof(uint8_t*));
  for(int c=0;c<M;++c) Y[c]=malloc((size_t)nb*Q8_STRIDE);
  float *sref=malloc((size_t)M*4), *s=malloc((size_t)M*4);
  unsigned sd=999;
  fill_q4(W,nb,&sd);
  for(int c=0;c<M;++c) fill_q8(Y[c],nb,&sd);

  gemm_a(K,M,sref,W,(const uint8_t*const*)Y);

  printf("# GEMM/prefill weight-decode-reuse probe  K=%d M(cols)=%d iters=%ld reps=%d\n",K,M,iters,reps);
  printf("# ns per (Mx output) (min of reps). ref s[0]=%.8e\n", sref[0]);

  struct { const char*name; gfn fn; } V[]={
    {"a_Mxvecdot (Mx vec_dot, PREFILL BASELINE)",        gemm_a},
    {"w_hoist    (decode weight once, reuse M cols)",    gemm_w_hoist},
  };
  for(size_t v=0;v<sizeof(V)/sizeof(V[0]);++v){
    memset(s,0,(size_t)M*4);
    V[v].fn(K,M,s,W,(const uint8_t*const*)Y);
    int ok=(memcmp(s,sref,(size_t)M*4)==0);
    double t=bench(V[v].fn,K,M,s,W,(const uint8_t*const*)Y,iters,reps);
    printf("  %-46s  %9.1f ns/Mout  %7.1f ns/col  gate=%s\n",
           V[v].name, t, t/M, ok?"PASS":"FAIL-BITEXACT");
  }
  double a=bench(gemm_a,K,M,s,W,(const uint8_t*const*)Y,iters,reps);
  double w=bench(gemm_w_hoist,K,M,s,W,(const uint8_t*const*)Y,iters,reps);
  printf("# speedup w_hoist vs a_Mxvecdot = %.3fx\n", a/w);

  for(int c=0;c<M;++c) free(Y[c]); free(Y); free(W); free(sref); free(s);
  return 0;
}
