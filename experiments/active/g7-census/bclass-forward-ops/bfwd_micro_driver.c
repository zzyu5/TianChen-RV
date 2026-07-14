/* bfwd_micro_driver.c — G7 L1 B-class forward-op kernel-sym symmetric micro A/B (hot/cold).
 * @k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / clang-18 symmetric.  [NG-4] kernel-axis, NOT e2e.
 *
 * 9 f32 forward ops: add mul scale cpy silu gelu rms_norm softmax rope.
 *   OURS = weft emitted RVV strip kernels (weft-opt --materialize-forward-elementwise-stream-front-door
 *          --lower-to-emitc | mlir-translate --mlir-to-cpp), compiled clang-18 -march symmetric.
 *   OPP  = ggml AS-SHIPPED: silu/softmax = EXPORTED stock libggml-cpu.so symbols; add/mul/scale/cpy/
 *          gelu/rms_norm/rope = verbatim ggml source (opponent_ggml.cpp), same clang-18 + march.
 *
 * ZERO-MODEL numeric gate: independent fp64 oracle recomputed from the SAME inputs.
 *   HARD bit-exact (0 ULP) : add mul scale cpy   (ours == oracle == opp, integer-ULP == 0)
 *   ULP-bounded            : silu softmax rms_norm rope (transcendental) ; gelu <= 1.3e-6 (tanhf)
 *
 * COLD protocol: pool of P input sets; total footprint >> k1 L2 (512KiB, NO L3) => every round streams
 *   each set cold from DRAM. P auto-sized from --target-mb. Per-set inputs AND outputs (read+write cold).
 * HOT: set 0 reused, best-of-passes.  Timing = CLOCK_MONOTONIC ns (k1 has NO cache-miss PMU;
 *   bandwidth = bytes_moved / time). N>=10 rounds, report median + relIQR.
 *
 * argv: <op> <n> <hot_iters> <rounds> <target_mb> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------------- OURS: weft emitted kernels ---------------- */
extern "C" void weft_emitc_vec_add_f32_kernel_vec_add_f32(size_t,const float*,const float*,float*);
extern "C" void weft_emitc_vec_mul_f32_kernel_vec_mul_f32(size_t,const float*,const float*,float*);
extern "C" void weft_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(size_t,float*,float);
extern "C" void weft_emitc_vec_cpy_f32_kernel_vec_cpy_f32(size_t,const float*,float*);
extern "C" void weft_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32(size_t,const float*,float*);
extern "C" void weft_emitc_gelu_f32_kernel_gelu_f32(size_t,const float*,float*);
extern "C" void weft_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(size_t,const float*,float*,float);
extern "C" double weft_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(size_t,float*,const float*,float);
extern "C" void weft_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(size_t,const float*,float*,float,float);

/* ---------------- OPP: ggml exported (stock .so) ---------------- */
extern "C" void   ggml_vec_silu_f32(const int n, float* y, const float* x);
extern "C" double ggml_vec_soft_max_f32(const int n, float* y, const float* x, float max);
/* ---------------- OPP: ggml verbatim (opponent_ggml.cpp) ---------------- */
extern "C" void opp_ggml_vec_add_f32(int n, float* z, const float* x, const float* y);
extern "C" void opp_ggml_vec_mul_f32(int n, float* z, const float* x, const float* y);
extern "C" void opp_ggml_vec_cpy_f32(int n, float* y, const float* x);
extern "C" void opp_ggml_vec_scale_f32(int n, float* y, const float v);
extern "C" void opp_ggml_vec_gelu_f32(int n, float* y, const float* x);
extern "C" void opp_gelu_init(void);
extern "C" void opp_ggml_rms_norm_f32(int n, const float* x, float* y, float eps);
extern "C" void opp_ggml_rope_norm_f32(int n, const float* x, float* y, float theta0, float theta_scale, float* cache);

enum Op { ADD,MUL,SCALE,CPY,SILU,GELU,RMSN,SOFTMAX,ROPE, NOP };
static const char* OPNAME[]={"add","mul","scale","cpy","silu","gelu","rms_norm","softmax","rope"};
static int parse_op(const char* s){ for(int i=0;i<NOP;i++) if(!strcmp(s,OPNAME[i])) return i; return -1; }
static int n_inbuf(int op){ return (op==ADD||op==MUL)?2:1; }        /* # input buffers per set */
static int inplace(int op){ return (op==SCALE); }                   /* op mutates the single buffer */

static const float EPS=1e-5f, THETA0=1.0f, THSCALE=0.99985f, SCALEV=0.7f, SMAX=1.25f;

/* bytes moved from DRAM per invocation (cold roofline denominator) */
static double bytes_moved(int op,int n){
    switch(op){
      case ADD: case MUL:      return 3.0*n*4;            /* 2 read + 1 write */
      case SCALE:              return 2.0*n*4;            /* read + write (in-place) */
      case CPY: case SILU: case GELU: case SOFTMAX: case ROPE: return 2.0*n*4; /* read + write */
      case RMSN:               return 3.0*n*4;            /* reduce-read + scale read + write */
      default: return 2.0*n*4;
    }
}

/* ---------------- rng + timing + stats ---------------- */
static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float rf(void){ return ((float)(xr()&0xffffff)/(float)0xffffff)*2.0f-1.0f; } /* [-1,1) */
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){ qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]); double q1=v[n/4],q3=v[(3*n)/4]; *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0; }

/* ---------------- dispatch: run ours / opp on (x0,x1,out) of length n ---------------- */
/* rope opponent needs a cache scratch; allocated by caller. */
static void run_ours(int op,int n,const float*x0,const float*x1,float*out){
    switch(op){
      case ADD:   weft_emitc_vec_add_f32_kernel_vec_add_f32((size_t)n,x0,x1,out); break;
      case MUL:   weft_emitc_vec_mul_f32_kernel_vec_mul_f32((size_t)n,x0,x1,out); break;
      case SCALE: /* in-place: caller pre-copies x0 into out */ weft_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32((size_t)n,out,SCALEV); break;
      case CPY:   weft_emitc_vec_cpy_f32_kernel_vec_cpy_f32((size_t)n,x0,out); break;
      case SILU:  weft_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32((size_t)n,x0,out); break;
      case GELU:  weft_emitc_gelu_f32_kernel_gelu_f32((size_t)n,x0,out); break;
      case RMSN:  weft_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32((size_t)n,x0,out,EPS); break;
      case SOFTMAX: weft_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32((size_t)n,out,x0,SMAX); break;
      case ROPE:  weft_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32((size_t)n,x0,out,THETA0,THSCALE); break;
    }
}
static void run_opp(int op,int n,const float*x0,const float*x1,float*out,float*cache){
    switch(op){
      case ADD:   opp_ggml_vec_add_f32(n,out,x0,x1); break;
      case MUL:   opp_ggml_vec_mul_f32(n,out,x0,x1); break;
      case SCALE: /* in-place: caller pre-copies x0 into out */ opp_ggml_vec_scale_f32(n,out,SCALEV); break;
      case CPY:   opp_ggml_vec_cpy_f32(n,out,x0); break;
      case SILU:  ggml_vec_silu_f32(n,out,x0); break;
      case GELU:  opp_ggml_vec_gelu_f32(n,out,x0); break;
      case RMSN:  opp_ggml_rms_norm_f32(n,x0,out,EPS); break;
      case SOFTMAX: (void)ggml_vec_soft_max_f32(n,out,x0,SMAX); break;
      case ROPE:  opp_ggml_rope_norm_f32(n,x0,out,THETA0,THSCALE,cache); break;
    }
}
/* fp64 ZERO-MODEL oracle -> f32 out */
static void oracle(int op,int n,const float*x0,const float*x1,float*out){
    switch(op){
      case ADD: for(int i=0;i<n;i++) out[i]=(float)((double)x0[i]+(double)x1[i]); break;
      case MUL: for(int i=0;i<n;i++) out[i]=(float)((double)x0[i]*(double)x1[i]); break;
      case SCALE: for(int i=0;i<n;i++) out[i]=(float)((double)x0[i]*(double)SCALEV); break;
      case CPY: for(int i=0;i<n;i++) out[i]=x0[i]; break;
      case SILU: for(int i=0;i<n;i++){ double x=x0[i]; out[i]=(float)(x/(1.0+exp(-x))); } break;
      case GELU: for(int i=0;i<n;i++){ double x=x0[i]; out[i]=(float)(0.5*x*(1.0+tanh(0.7978845608028654*x*(1.0+0.044715*x*x)))); } break;
      case RMSN:{ double s=0; for(int i=0;i<n;i++) s+=(double)x0[i]*(double)x0[i]; float mean=(float)(s/(double)n); float sc=1.0f/sqrtf(mean+EPS); for(int i=0;i<n;i++) out[i]=(float)((double)x0[i]*(double)sc); } break;
      case SOFTMAX: for(int i=0;i<n;i++) out[i]=(float)exp((double)x0[i]-(double)SMAX); break;
      case ROPE:{ double th=THETA0; for(int i=0;i<n/2;i++){ double c=cos(th),sn=sin(th); double a=x0[2*i],b=x0[2*i+1]; out[2*i]=(float)(a*c-b*sn); out[2*i+1]=(float)(a*sn+b*c); th*=THSCALE; } } break;
    }
}
static uint32_t ulp_diff(float a,float b){ if(a==b) return 0; uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4);
    if((ua>>31)!=(ub>>31)) return 0xffffffu; /* opposite signs (both ~0 handled by a==b) */
    return ua>ub?ua-ub:ub-ua; }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s op n hot_iters rounds target_mb seed\n",argv[0]); return 2; }
    int op=parse_op(argv[1]); if(op<0){ fprintf(stderr,"bad op\n"); return 2; }
    int n=atoi(argv[2]), hiters=atoi(argv[3]), rounds=atoi(argv[4]); double tmb=atof(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    if(n<32) n=32; if(n&1) n++; if(hiters<1)hiters=1; if(rounds<8)rounds=8; if(tmb<1)tmb=4;
    if(op==GELU) opp_gelu_init();

    int nin=n_inbuf(op);
    /* per-set bytes = nin inputs + 1 output (scale: only the mutated buffer) */
    size_t set_in = (size_t)nin*n*sizeof(float);
    size_t set_out = inplace(op)?0:(size_t)n*sizeof(float);
    size_t set_bytes = set_in + set_out;
    int P = (int)((tmb*1e6)/(double)set_bytes); if(P<8)P=8; if(P>4096)P=4096;

    /* pool */
    float **X0=(float**)malloc(P*sizeof(float*)), **X1=(float**)malloc(P*sizeof(float*));
    float **OU=(float**)malloc(P*sizeof(float*)), **OO=(float**)malloc(P*sizeof(float*));
    float *cache=(float*)aligned_alloc(64,(size_t)n*sizeof(float)); /* rope opp scratch (shared, warm) */
    if(!X0||!X1||!OU||!OO||!cache){ fprintf(stderr,"OOM meta\n"); return 3; }
    for(int p=0;p<P;p++){
        X0[p]=(float*)aligned_alloc(64,(size_t)n*sizeof(float));
        X1[p]=(nin==2)?(float*)aligned_alloc(64,(size_t)n*sizeof(float)):NULL;
        OU[p]=(float*)aligned_alloc(64,(size_t)n*sizeof(float));
        OO[p]=(float*)aligned_alloc(64,(size_t)n*sizeof(float));
        if(!X0[p]||!OU[p]||!OO[p]||(nin==2&&!X1[p])){ fprintf(stderr,"OOM pool %d\n",p); return 3; }
        for(int i=0;i<n;i++){ X0[p][i]=rf(); if(nin==2) X1[p][i]=rf(); }
    }

    /* ---- ZERO-MODEL numeric gate on set 0 ---- */
    float* orc=(float*)aligned_alloc(64,(size_t)n*sizeof(float));
    oracle(op,n,X0[0],X1[0],orc);
    if(inplace(op)){ memcpy(OU[0],X0[0],(size_t)n*sizeof(float)); memcpy(OO[0],X0[0],(size_t)n*sizeof(float)); }
    run_ours(op,n,X0[0],X1[0],OU[0]);
    run_opp (op,n,X0[0],X1[0],OO[0],cache);
    uint32_t maxulp_ours=0, maxulp_opp=0, maxulp_oo=0; double maxrel_ours=0, maxrel_opp=0; int nbad=0;
    for(int i=0;i<n;i++){
        uint32_t uo=ulp_diff(OU[0][i],orc[i]), up=ulp_diff(OO[0][i],orc[i]), uc=ulp_diff(OU[0][i],OO[0][i]);
        if(uo>maxulp_ours)maxulp_ours=uo; if(up>maxulp_opp)maxulp_opp=up; if(uc>maxulp_oo)maxulp_oo=uc;
        double den=fabs((double)orc[i])>1e-6?fabs((double)orc[i]):1e-6;
        double ro=fabs((double)OU[0][i]-(double)orc[i])/den, rp=fabs((double)OO[0][i]-(double)orc[i])/den;
        if(ro>maxrel_ours)maxrel_ours=ro; if(rp>maxrel_opp)maxrel_opp=rp;
        if(ro>1.5e-3)nbad++;
    }
    int hard_exact = (op==ADD||op==MUL||op==SCALE||op==CPY);
    int gate;
    if(hard_exact)        gate = (maxulp_ours==0 && maxulp_oo==0);         /* 0-ULP vs oracle AND vs opp */
    else if(op==GELU)     gate = (maxrel_ours<=1.3e-6);
    else                  gate = (maxrel_ours<=2e-4);                       /* ULP-bounded transcendental */

    /* ---- HOT: set 0 reused, best-of-passes ---- */
    volatile double sink=0;
    for(int w=0;w<3;w++){ if(inplace(op))memcpy(OU[0],X0[0],(size_t)n*sizeof(float)); run_ours(op,n,X0[0],X1[0],OU[0]);
                          if(inplace(op))memcpy(OO[0],X0[0],(size_t)n*sizeof(float)); run_opp(op,n,X0[0],X1[0],OO[0],cache); }
    double ours_hot=1e30, opp_hot=1e30;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns(); for(int it=0;it<hiters;it++){ run_ours(op,n,X0[0],X1[0],OU[0]); } double t1=now_ns();
        double npc=(t1-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; sink+=OU[0][0];
        double t2=now_ns(); for(int it=0;it<hiters;it++){ run_opp(op,n,X0[0],X1[0],OO[0],cache); } double t3=now_ns();
        npc=(t3-t2)/(double)hiters; if(npc<opp_hot)opp_hot=npc; sink+=OO[0][0];
    }
    double bmv=bytes_moved(op,n);
    printf("HOT op=%s n=%d P=%d GATE=%s maxulp_ours=%u maxulp_opp=%u maxulp_ours_vs_opp=%u "
           "relerr_ours=%.3e relerr_opp=%.3e nbad=%d ours_ns=%.1f opp_ns=%.1f "
           "ours_GBs=%.3f opp_GBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           OPNAME[op],n,P,gate?"PASS":"FAIL",maxulp_ours,maxulp_opp,maxulp_oo,
           maxrel_ours,maxrel_opp,nbad,ours_hot,opp_hot,bmv/ours_hot,bmv/opp_hot,
           opp_hot/ours_hot,(double)sink);
    /* ratio_ours_over_opp = (ours GB/s)/(opp GB/s) = opp_ns/ours_ns => >1 means ours faster */

    /* ---- COLD: pool median-of-rounds (inputs+outputs streamed cold; k1 L2=512KiB, no L3) ---- */
    for(int w=0;w<2;w++){ for(int p=0;p<P;p++){ if(inplace(op))memcpy(OU[p],X0[p],(size_t)n*sizeof(float)); run_ours(op,n,X0[p],X1[p],OU[p]); }
                          for(int p=0;p<P;p++){ if(inplace(op))memcpy(OO[p],X0[p],(size_t)n*sizeof(float)); run_opp(op,n,X0[p],X1[p],OO[p],cache); } sink+=OU[0][0]+OO[0][0]; }
    double *to=(double*)malloc(rounds*sizeof(double)), *tp=(double*)malloc(rounds*sizeof(double));
    for(int rd=0;rd<rounds;rd++){
        double t0=now_ns(); for(int p=0;p<P;p++){ if(inplace(op))memcpy(OU[p],X0[p],(size_t)n*sizeof(float)); run_ours(op,n,X0[p],X1[p],OU[p]); } double t1=now_ns(); to[rd]=(t1-t0)/(double)P; sink+=OU[0][0];
        double t2=now_ns(); for(int p=0;p<P;p++){ if(inplace(op))memcpy(OO[p],X0[p],(size_t)n*sizeof(float)); run_opp(op,n,X0[p],X1[p],OO[p],cache); } double t3=now_ns(); tp[rd]=(t3-t2)/(double)P; sink+=OO[0][0];
    }
    double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
    double totmb=(double)P*set_bytes/1e6;
    printf("COLD op=%s n=%d P=%d ws_MB=%.1f ours_ns_med=%.1f ours_iqrpct=%.2f ours_GBs=%.3f "
           "opp_ns_med=%.1f opp_iqrpct=%.2f opp_GBs=%.3f ratio_ours_over_opp=%.4f sink=%.1f\n",
           OPNAME[op],n,P,totmb,om,oi,bmv/om,pm,pi,bmv/pm,pm/om,(double)sink);
    /* ratio = opp_ns/ours_ns = (ours GB/s)/(opp GB/s); >1 ours faster, <1 ours slower */
    return gate?0:1;
}
