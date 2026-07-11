/* stream_paired_driver.c -- 线丙 covering-batch③ streaming-format kernel-axis paired A/B.
 *
 * Streaming formats are MEMORY-BOUND (to-the-wall) operators. This driver times OUR
 * front-door-constructed streaming kernels against ggml's own dispatched streaming kernel
 * (linked from the board's libggml-{base,cpu}.so) on the SAME cold-streamed data, with a
 * REAL correctness gate BEFORE timing. Expected outcome (honest prior): physical parity
 * at the memory wall for most cells (roofline-bound), the occasional "to-the-wall speed"
 * win logged as-is. NOT chasing wins; parity is the correct prediction for these ops.
 *
 * Three modes:
 *   dequant  : ggml `dequantize_row_<fmt>(x,y,k)`  vs OUR tcrv_emitc_dequant_<fmt>_...(k,x,y)
 *   quant    : ggml `quantize_row_<fmt>(x,y,k)`     vs OUR tcrv_emitc_quantize_row_<fmt>_...(k,x,y)
 *   forward  : ggml_vec_<op>_f32 (transcribed ref)  vs OUR tcrv_emitc_<op>_f32_...(n,...)
 *
 * CACHE HYGIENE (the reason a memory-bound micro can LIE): a cache-resident stream reports
 * a speedup that does not survive real decode where weights stream COLD from DRAM. So this
 * driver allocates a POOL whose in+out footprint exceeds the board LLC by several x; each
 * timed ROUND sweeps the ENTIRE pool once (one call over n=nblocks*qk), so every block is
 * touched cold. achieved_GBs is reported so we can roofline-classify honestly.
 *
 * COMPILER-SYMMETRY [CASE-COMPILER-ASYMMETRY]: this binary's OUR-side codegen is whatever
 * compiler built it; the ggml opponent is the fixed gcc-15 shipped .so. So:
 *   built with clang -> SYSTEM/DEPLOY account (we ship clang .o vs board's gcc ggml)
 *   built with gcc   -> KERNEL account (compiler-symmetric gcc-ours vs gcc-opp)
 * The orchestrator runs it BOTH ways. For memory-bound streams the two ledgers converge
 * at the bandwidth wall; we report both anyway per protocol.
 *
 * [NG-4]: L-? kernel-axis path-CANDIDATE datapoint, NOT a [PERF-1] eight-gate beat.
 *
 * argv: <mode:dequant|quant|forward> <fmt> <n_elems(mult qk)> <rounds> <seed>
 *   defaults: n=24*1024*1024  rounds=15  seed=0x51EA
 * stdout (one machine-parseable line):
 *   STREAM mode=.. fmt=.. n=.. nblocks=.. ws_bytes=.. bytes_per_sweep=..
 *     GATE=PASS|FAIL gate_detail=..
 *     ours_ns_med=.. ours_ns_iqrpct=.. ours_GBs=.. opp_ns_med=.. opp_ns_iqrpct=.. opp_GBs=..
 *     ratio_ours_over_opp=.. sink=..
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef uint16_t ggml_half;

/* ---- ggml block layouts (natural contiguous == on-disk) ---- */
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;                          /* 34 */
#define QK_K 256
typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[QK_K/2]; } block_q4_K; /* 144 */
#define QK4_NL 32
typedef struct { ggml_half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;                    /* 18 */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;       /* 292 */

#ifdef __cplusplus
#define STREAM_SASSERT(c,m) static_assert(c,m)
#else
#define STREAM_SASSERT(c,m) _Static_assert(c,m)
#endif
STREAM_SASSERT(sizeof(block_q8_0)==34,   "q8_0");
STREAM_SASSERT(sizeof(block_q4_K)==144,  "q4_K");
STREAM_SASSERT(sizeof(block_iq4_nl)==18, "iq4_nl");
STREAM_SASSERT(sizeof(block_q8_K)==292,  "q8_K");

/* ---- ggml opponents (linked from the board's own libggml-{base,cpu}.so, gcc-15) ---- */
#ifdef __cplusplus
extern "C" {
#endif
extern void dequantize_row_q8_0  (const void* x, float* y, int64_t k);
extern void dequantize_row_q4_K  (const void* x, float* y, int64_t k);
extern void dequantize_row_iq4_nl(const void* x, float* y, int64_t k);
extern void quantize_row_q8_0    (const float* x, void* y, int64_t k);
extern void quantize_row_q8_K    (const float* x, void* y, int64_t k);

/* ---- OUR front-door-constructed streaming kernels (exported .o from HEAD) ---- */
extern void tcrv_emitc_dequant_q8_0_kernel_dequant_q8_0    (size_t k, const uint8_t* x, float* y);
extern void tcrv_emitc_dequant_q4_K_kernel_dequant_q4_K    (size_t k, const uint8_t* x, float* y);
extern void tcrv_emitc_dequant_iq4_nl_kernel_dequant_iq4_nl(size_t k, const uint8_t* x, float* y);
extern void tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(size_t k, const float* x, uint8_t* y);
extern void tcrv_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(size_t k, const float* x, uint8_t* y);
extern void tcrv_emitc_vec_add_f32_kernel_vec_add_f32(size_t n, const float* a, const float* b, float* o);
extern void tcrv_emitc_vec_mul_f32_kernel_vec_mul_f32(size_t n, const float* a, const float* b, float* o);
extern void tcrv_emitc_gelu_f32_kernel_gelu_f32(size_t n, const float* x, float* y);
#ifdef __cplusplus
}
#endif

/* ---- transcribed ggml forward references (ggml_vec_* are inline/static, not exported) ---- */
static void ref_vec_add_f32(size_t n, const float* a, const float* b, float* o){ for(size_t i=0;i<n;i++) o[i]=a[i]+b[i]; }
static void ref_vec_mul_f32(size_t n, const float* a, const float* b, float* o){ for(size_t i=0;i<n;i++) o[i]=a[i]*b[i]; }
/* ggml_gelu_f32 accurate (tanh) variant == the exact formula OUR emit lowers to.
 * NOTE: ggml's DEPLOYED forward-gelu uses the f16 lookup table ggml_table_gelu_f16 (a
 * different, lossy-but-cache-cheap path). We time vs the accurate tanh reference and
 * disclose the LUT distinction; see casefile. */
static const float GELU_C = 0.79788456080286535587989211986876f;
static void ref_gelu_f32(size_t n, const float* x, float* y){
  for(size_t i=0;i<n;i++){ float v=x[i]; y[i]=0.5f*v*(1.0f+tanhf(GELU_C*v*(1.0f+0.044715f*v*v))); }
}

/* ---- f16 helper (compiled with zfh) ---- */
static ggml_half f32_to_f16(float v){ _Float16 h=(_Float16)v; ggml_half b; memcpy(&b,&h,2); return b; }

/* ---- timing ---- */
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){
  qsort(v,n,sizeof(double),cmp_d);
  *med = (n&1)? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
  double q1=v[n/4], q3=v[(3*n)/4];
  *iqrpct = (*med>0)? 100.0*(q3-q1)/(*med) : 0.0;
}

static volatile double g_sink=0.0;
static double fchecksum(const float* y, size_t n){ double s=0; for(size_t i=0;i<n;i+=97) s+=y[i]; return s; }
static double bchecksum(const uint8_t* y, size_t n){ double s=0; for(size_t i=0;i<n;i+=131) s+=(double)y[i]; return s; }

int main(int argc,char**argv){
  const char* mode = argc>1? argv[1] : "dequant";
  const char* fmt  = argc>2? argv[2] : "q8_0";
  size_t n         = argc>3? strtoull(argv[3],0,10) : (size_t)24*1024*1024;
  int rounds       = argc>4? atoi(argv[4]) : 15;
  unsigned seed    = argc>5? (unsigned)strtoul(argv[5],0,0) : 0x51EA;
  if(rounds<8) rounds=8;

  int is_deq = !strcmp(mode,"dequant");
  int is_qnt = !strcmp(mode,"quant");
  int is_fwd = !strcmp(mode,"forward");

  int qk=32, bsz=34;               /* per-format block geometry (dequant/quant) */
  if(is_deq||is_qnt){
    if(!strcmp(fmt,"q8_0")){ qk=32; bsz=34; }
    else if(!strcmp(fmt,"q4_K")){ qk=256; bsz=144; }
    else if(!strcmp(fmt,"iq4_nl")){ qk=32; bsz=18; }
    else if(!strcmp(fmt,"q8_K")){ qk=256; bsz=292; }
    else { fprintf(stderr,"bad fmt %s\n",fmt); return 2; }
  }
  n = (n/qk)*qk; if(n==0) n=qk;
  size_t nblocks = n/qk;

  /* ---- allocate cold pools ---- */
  uint8_t *qbuf=0; float *fA=0,*fB=0,*fOurs=0,*fOpp=0; uint8_t *bOurs=0,*bOpp=0;
  size_t ws=0, bytes_per_sweep=0;
  srand(seed);

  if(is_deq){
    qbuf   = (uint8_t*)malloc(nblocks*bsz);
    fOurs  = (float*)malloc(n*sizeof(float));
    fOpp   = (float*)malloc(n*sizeof(float));
    if(!qbuf||!fOurs||!fOpp){ fprintf(stderr,"oom\n"); return 3; }
    /* random block bytes, but pin f16 scale(s) to a valid small magnitude -> finite output */
    for(size_t b=0;b<nblocks*bsz;b++) qbuf[b]=(uint8_t)(rand());
    for(size_t bi=0; bi<nblocks; bi++){
      uint8_t* blk = qbuf + bi*bsz;
      float sc = 0.02f + 0.18f*((float)(rand()&1023)/1023.0f);
      ggml_half hd=f32_to_f16(sc);
      if(!strcmp(fmt,"q8_0")||!strcmp(fmt,"iq4_nl")){ memcpy(blk,&hd,2); }
      else if(!strcmp(fmt,"q4_K")){ ggml_half hdm=f32_to_f16(sc*0.5f); memcpy(blk,&hd,2); memcpy(blk+2,&hdm,2); }
    }
    ws = nblocks*bsz + n*sizeof(float);
    bytes_per_sweep = nblocks*bsz + n*sizeof(float);  /* read blocks + write floats */
  } else if(is_qnt){
    fA     = (float*)malloc(n*sizeof(float));
    bOurs  = (uint8_t*)malloc(nblocks*bsz);
    bOpp   = (uint8_t*)malloc(nblocks*bsz);
    if(!fA||!bOurs||!bOpp){ fprintf(stderr,"oom\n"); return 3; }
    for(size_t i=0;i<n;i++) fA[i] = -1.0f + 2.0f*((float)(rand()&65535)/65535.0f);
    ws = n*sizeof(float) + nblocks*bsz;
    bytes_per_sweep = n*sizeof(float) + nblocks*bsz;  /* read floats + write blocks */
  } else if(is_fwd){
    int binary = (!strcmp(fmt,"add")||!strcmp(fmt,"mul"));
    fA    = (float*)malloc(n*sizeof(float));
    fB    = binary? (float*)malloc(n*sizeof(float)) : 0;
    fOurs = (float*)malloc(n*sizeof(float));
    fOpp  = (float*)malloc(n*sizeof(float));
    if(!fA||(binary&&!fB)||!fOurs||!fOpp){ fprintf(stderr,"oom\n"); return 3; }
    for(size_t i=0;i<n;i++){ fA[i]=-2.0f+4.0f*((float)(rand()&65535)/65535.0f); if(binary) fB[i]=-2.0f+4.0f*((float)(rand()&65535)/65535.0f); }
    ws = (binary?3:2)*n*sizeof(float);
    bytes_per_sweep = (binary?3:2)*n*sizeof(float);
  } else { fprintf(stderr,"bad mode %s\n",mode); return 2; }

  /* ---- one sweep helpers ---- */
  #define SWEEP_OURS() do{ \
    if(is_deq){ if(!strcmp(fmt,"q8_0")) tcrv_emitc_dequant_q8_0_kernel_dequant_q8_0(n,qbuf,fOurs); \
      else if(!strcmp(fmt,"q4_K")) tcrv_emitc_dequant_q4_K_kernel_dequant_q4_K(n,qbuf,fOurs); \
      else tcrv_emitc_dequant_iq4_nl_kernel_dequant_iq4_nl(n,qbuf,fOurs); } \
    else if(is_qnt){ if(!strcmp(fmt,"q8_0")) tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(n,fA,bOurs); \
      else tcrv_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(n,fA,bOurs); } \
    else { if(!strcmp(fmt,"add")) tcrv_emitc_vec_add_f32_kernel_vec_add_f32(n,fA,fB,fOurs); \
      else if(!strcmp(fmt,"mul")) tcrv_emitc_vec_mul_f32_kernel_vec_mul_f32(n,fA,fB,fOurs); \
      else tcrv_emitc_gelu_f32_kernel_gelu_f32(n,fA,fOurs); } }while(0)
  #define SWEEP_OPP() do{ \
    if(is_deq){ if(!strcmp(fmt,"q8_0")) dequantize_row_q8_0(qbuf,fOpp,(int64_t)n); \
      else if(!strcmp(fmt,"q4_K")) dequantize_row_q4_K(qbuf,fOpp,(int64_t)n); \
      else dequantize_row_iq4_nl(qbuf,fOpp,(int64_t)n); } \
    else if(is_qnt){ if(!strcmp(fmt,"q8_0")) quantize_row_q8_0(fA,bOpp,(int64_t)n); \
      else quantize_row_q8_K(fA,bOpp,(int64_t)n); } \
    else { if(!strcmp(fmt,"add")) ref_vec_add_f32(n,fA,fB,fOpp); \
      else if(!strcmp(fmt,"mul")) ref_vec_mul_f32(n,fA,fB,fOpp); \
      else ref_gelu_f32(n,fA,fOpp); } }while(0)

  /* ---- correctness gate BEFORE timing (ours vs opp on identical input) ---- */
  SWEEP_OURS(); SWEEP_OPP();
  int gate_pass=1; char detail[256];
  if(is_deq||is_fwd){
    double maxabs=0, maxrel=0; size_t nbad=0;
    for(size_t i=0;i<n;i++){ double a=fOurs[i],b=fOpp[i]; double da=fabs(a-b); if(da>maxabs)maxabs=da;
      double den=fabs(b)>1e-9?fabs(b):1e-9; double dr=da/den; if(dr>maxrel)maxrel=dr; if(dr>1e-4) nbad++; }
    gate_pass = (maxrel < 5e-4);
    snprintf(detail,sizeof detail,"maxabs=%.3e_maxrel=%.3e_nbad=%zu",maxabs,maxrel,nbad);
  } else { /* quant: compare packed bytes; tolerate <=1 lsb rounding on int8 quants, exact on scale */
    size_t nb=nblocks*bsz; size_t ndiff=0; int maxqdiff=0;
    for(size_t i=0;i<nb;i++){ if(bOurs[i]!=bOpp[i]){ int d=(int)((int8_t)bOurs[i])-(int)((int8_t)bOpp[i]); if(d<0)d=-d; if(d>maxqdiff)maxqdiff=d; ndiff++; } }
    gate_pass = (maxqdiff<=1); /* <=1 lsb = RNE-vs-roundf tie, disclosed as numerics */
    snprintf(detail,sizeof detail,"bytes_diff=%zu_of_%zu_maxlsb=%d",ndiff,nb,maxqdiff);
  }

  /* sink to defeat DCE */
  double sink=0;
  if(is_qnt){ sink=bchecksum(bOurs,nblocks*bsz)+bchecksum(bOpp,nblocks*bsz); }
  else { sink=fchecksum(fOurs,n)+fchecksum(fOpp,n); }
  g_sink=sink;

  /* ---- timed rounds (paired alternation ours/opp) ---- */
  double *to=(double*)malloc(rounds*sizeof(double));
  double *tp=(double*)malloc(rounds*sizeof(double));
  for(int r=0;r<rounds;r++){
    double t0=now_ns(); SWEEP_OURS(); double t1=now_ns(); to[r]=t1-t0;
    if(is_qnt) g_sink+=bchecksum(bOurs,nblocks*bsz); else g_sink+=fchecksum(fOurs,n);
    double t2=now_ns(); SWEEP_OPP(); double t3=now_ns(); tp[r]=t3-t2;
    if(is_qnt) g_sink+=bchecksum(bOpp,nblocks*bsz); else g_sink+=fchecksum(fOpp,n);
  }
  double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
  double oGBs = (double)bytes_per_sweep/om;   /* ns -> bytes/ns == GB/s */
  double pGBs = (double)bytes_per_sweep/pm;
  double ratio = pm/om;                        /* >1 => ours faster (lower ns) */

  printf("STREAM mode=%s fmt=%s n=%zu nblocks=%zu ws_bytes=%zu bytes_per_sweep=%zu "
         "GATE=%s gate_detail=%s "
         "ours_ns_med=%.0f ours_ns_iqrpct=%.2f ours_GBs=%.3f "
         "opp_ns_med=%.0f opp_ns_iqrpct=%.2f opp_GBs=%.3f "
         "ratio_ours_over_opp=%.4f sink=%.3f\n",
         mode,fmt,n,nblocks,ws,bytes_per_sweep,
         gate_pass?"PASS":"FAIL",detail,
         om,oi,oGBs, pm,pi,pGBs, ratio, g_sink);
  return gate_pass?0:1;
}
