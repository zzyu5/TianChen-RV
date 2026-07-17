// P2 decode-debt: q4_1 / q8_0 DECODE (M=1 GEVM) paired A/B driver.
//
// PROVENANCE: this is A2-batch4-gemm-decode-M1-raw/flat_gevm_m1_driver.cpp EXTENDED with the
// two formats batch4 §4 listed as BLOCKED-ON-CONSTRUCTION (q4_1 / q8_0 @ both boards). The
// batch4 blocker ("no clean weft-opt leaf source; dataflow uses the retired monolithic
// repack_gemv_q4_1_q8_1; lower-to-emitc rejects exec.variant") is STALE: the front door now
// carries kNibbleQ41ScaleModel ("dual-fp16-per-block-d_x.d_y-plus-min") and kNibbleQ80ScaleModel
// ("dual-fp16-per-block-d_x.d_y-full-i8") and CONSTRUCTS typed_repack_gemv_loop_body from an
// m_regime="decode" quant_contraction request. Both leaves here are FRONT-DOOR LIVE EXPORTS
// (see GEN_SEAL.txt) -- no retired monolithic emitter, no archived .inc.
//
//  OURS  = repack-GEVM LEAF: ONE GEVM call over nc columns, x16-interleaved weight + PLAIN
//          single q8 activation vector (M=1). Leaf symbol contains "vec_dot" but its BODY is
//          typed_repack_gemv_loop_body -- NOT the block-dot vec_dot path (batch4 §0 key
//          distinction; discriminator = kernel body shape, not symbol name).
//  OPP-X = the board's REAL dispatched block-dot ggml_vec_dot_<fmt> (stock libggml-cpu.so),
//          per-column (nrc=1) over the SAME plain blocks == block-dot @ M=1. This is a
//          CROSS-OPERATOR comparison (our GEVM vs their per-column block-dot) -- the opponent
//          recon names for both cells. NOT a same-operator hard win.
//  OPP-S = (q8_0 ONLY) ggml_gemv_q8_0_16x1_q8_0 -- the stock SAME-OPERATOR repack GEVM
//          (271 insns / 171 vector on rvv = a real heavy hand-tuned kernel, NOT a thunk).
//          recon does NOT name it; per the P1-backfill7 iq4_nl OPP-S precedent a same-op
//          opponent must be reported when it exists. We feed OPP-S *our* packed buffer and let
//          the ZERO-MODEL ORACLE decide whether ggml's 16x1 layout is byte-compatible with
//          ours: oracle-clean => the OPP-S number is valid; oracle-mismatch => layouts differ
//          and OPP-S is reported UNMEASURABLE (never silently timed against a wrong layout).
//          q4_1 has NO ggml_gemv_q4_1_* symbol at all => CROSSOP is its only legal opponent.
//  GATE  = ZERO-MODEL independent scalar oracle recomputed from PLAIN inputs, per column.
//  ratio_cold = opp_med / ours_med   (>=0.8 => PASS ; <0.8 => named-X).
// cold: 32MiB flush, CLOCK_MONOTONIC, median+relIQR, core-pinned.
//  argv: <fmt q4_1|q8_0> K(mult32) N(nc,mult16) reps [verify_only] [seed]
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <time.h>

typedef _Float16 ggml_half;

// ---------------- block layouts ----------------
#define QK 32
struct block_q4_1 { ggml_half d; ggml_half m; uint8_t qs[16]; };        // 20
struct block_q8_0 { ggml_half d; int8_t qs[32]; };                      // 34
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[32]; };         // 36
// x16 layouts READ OFF THE GENERATED LEAF (not guessed):
//  q4_1 leaf: group base g*nb*320; weight qs addr = 64 + j*16 (+8 = 2nd 8-lane strip), j<16;
//             d strip @0/+16, m strip @32/+48; act q8_1 qs @4 (lo) and @20 (hi), d@0, s@2.
//  q8_0 leaf: group base g*nb*544; weight qs addr = 32 + p*16 (+8), p<32; d strip @0/+16;
//             act q8_0 qs @2, d@0.
struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; };  // 320
struct block_q8_0x16 { ggml_half d[16]; int8_t qs[512]; };                    // 544
static_assert(sizeof(block_q4_1)==20,"q4_1"); static_assert(sizeof(block_q8_0)==34,"q8_0");
static_assert(sizeof(block_q8_1)==36,"q8_1");
static_assert(sizeof(block_q4_1x16)==320,"q4_1x16");
static_assert(sizeof(block_q8_0x16)==544,"q8_0x16");

// ---------------- OURS front-door repack-GEVM leaf externs (8-arg vec_dot ABI, nc@arg3) ----
// ONE leaf .o is linked per binary => the extern for the OTHER format must not be referenced
// (an unguarded extern + dispatch makes lld fail on an undefined symbol). Exactly one of
// FMT_Q4_1 / FMT_Q8_0 is defined by the runner.
#if !defined(FMT_Q4_1) && !defined(FMT_Q8_0)
#error "define exactly one of FMT_Q4_1 / FMT_Q8_0"
#endif
#ifdef FMT_Q4_1
extern "C" void weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
static inline void our_q4_1(size_t K,float*o,const uint8_t*p,const uint8_t*a,size_t nc){
    weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(K,o,nc,p,0,a,0,0);}
#endif
#ifdef FMT_Q8_0
extern "C" void weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
static inline void our_q8_0(size_t K,float*o,const uint8_t*p,const uint8_t*a,size_t nc){
    weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(K,o,nc,p,0,a,0,0);}
#endif

// ---------------- OPP-X stock-.so block-dot (CROSSOP) ----------------
extern "C" void ggml_vec_dot_q4_1_q8_1(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);
// ---------------- OPP-S stock same-operator repack GEVM (q8_0 only) ----------------
extern "C" void ggml_gemv_q8_0_16x1_q8_0(int n,float*s,size_t bs,const void*vx,const void*vy,int nr,int nc);

// ---------------- x16 repack (byte-exact to the leaf addressing above) ----------------
// q4_1: RAW unsigned nibble decode (front door stamps weight_nibble_unsigned, [0,15], NO
// offset-binary -8) => NO ^0x88 XOR, unlike q4_0's pack. The asymmetric bias lives in the
// separate per-block MIN scale (m_x * s_y), which is why q4_1 must not borrow q4_0's XOR.
static block_q4_1x16 pack_q4_1(block_q4_1* in){ block_q4_1x16 o;
    for(int c=0;c<16;c++){o.d[c]=in[c].d;o.m[c]=in[c].m;}
    for(int j=0;j<16;j++)for(int c=0;c<16;c++)o.qs[j*16+c]=in[c].qs[j];
    return o; }
// q8_0: full int8, 32 positions x 16 columns.
static block_q8_0x16 pack_q8_0(block_q8_0* in){ block_q8_0x16 o;
    for(int c=0;c<16;c++)o.d[c]=in[c].d;
    for(int p=0;p<32;p++)for(int c=0;c<16;c++)o.qs[p*16+c]=in[c].qs[p];
    return o; }

// ---------------- ZERO-MODEL scalar oracles (recomputed from PLAIN blocks) ----------------
static float ora_q4_1(const block_q4_1&w,const block_q8_1&a){ int s=0;
    for(int j=0;j<16;j++){ int x0=w.qs[j]&0xF, x1=w.qs[j]>>4;      // RAW unsigned [0,15]
        s+=x0*a.qs[j]+x1*a.qs[j+16]; }
    return (float)w.d*(float)a.d*(float)s+(float)w.m*(float)a.s; }
static float ora_q8_0(const block_q8_0&w,const block_q8_0&a){ int s=0;
    for(int j=0;j<32;j++) s+=w.qs[j]*a.qs[j];
    return (float)s*(float)w.d*(float)a.d; }

// ---------------- cold timing infra ----------------
static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
#ifndef FLUSH_MB
#define FLUSH_MB 32
#endif
static const size_t FLUSH_BYTES=(size_t)FLUSH_MB*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&&f,double*out,int reps){ for(int r=0;r<reps;r++){ cold_flush(); double a=now_ms(); f(); out[r]=now_ms()-a; } }
static double med(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

// ---------------- per-format test ----------------
// INJECT: anti-hollow falsifiers. 0=clean, 1=corrupt oracle input, 2=corrupt DUT output.
#ifndef INJECT
#define INJECT 0
#endif
template<class BW,class BA,class BP>
static int run_fmt(const char* tag,int N,int K,bool do_perf,int reps,unsigned seed,
                   void(*fillW)(std::vector<BW>&,int,std::mt19937&),
                   void(*fillA)(std::vector<BA>&,int,std::mt19937&),
                   BP(*pack)(BW*),
                   void(*ourfn)(size_t,float*,const uint8_t*,const uint8_t*,size_t),
                   void(*oppfn)(int,float*,size_t,const void*,size_t,const void*,size_t,int),
                   float(*orafn)(const BW&,const BA&), double reltol,
                   bool has_opps,
                   void(*oppsfn)(int,float*,size_t,const void*,const void*,int,int)){
    int nb=K/32; std::mt19937 rng(seed);
    std::vector<BW> W((size_t)N*nb); fillW(W,nb,rng);
    std::vector<BA> act(nb); fillA(act,nb,rng);
    int ng=N/16; std::vector<BP> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++)for(int b=0;b<nb;b++){ BW tmp[16]; for(int c=0;c<16;c++)tmp[c]=W[(size_t)(g*16+c)*nb+b]; packed[(size_t)g*nb+b]=pack(tmp); }

    std::vector<float> ours(N,0.f), ropp(N,0.f), rora(N,0.f), ropps(N,0.f);
    ourfn((size_t)K,ours.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N);
#if INJECT==2
    ours[N/2]+=1.0f;   // DUT-output fault: OURS must go RED, oracle/opp stay GREEN
#endif
    for(int r=0;r<N;r++){ oppfn(K,&ropp[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1);
        double s=0; for(int b=0;b<nb;b++){ BW wb=W[(size_t)r*nb+b];
#if INJECT==1
            if(r==N/2&&b==0) wb.qs[0]^=0x01;   // oracle-input fault: ALL must go RED
#endif
            s+=orafn(wb,act[b]); } rora[r]=(float)s; }
    // OPP-S same-operator GEVM over OUR packed buffer; oracle decides layout compatibility.
    int opps_mism=-1; float opps_maxrel=0.f;
    if(has_opps&&oppsfn){ oppsfn(K,ropps.data(),0,(const void*)packed.data(),(const void*)act.data(),1,N);
        opps_mism=0; for(int r=0;r<N;r++){ float a=std::fabs(ropps[r]-rora[r])/(std::fabs(rora[r])+1e-6f);
            if(a>=reltol)opps_mism++; if(a>opps_maxrel)opps_maxrel=a; } }
    int m_opp=0,m_ora=0; float mr_opp=0,mr_ora=0;
    for(int r=0;r<N;r++){
        float ao=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f);
        float aq=std::fabs(ours[r]-rora[r])/(std::fabs(rora[r])+1e-6f);
        if(ao>=reltol)m_opp++; if(aq>=reltol)m_ora++;
        if(ao>mr_opp)mr_opp=ao; if(aq>mr_ora)mr_ora=aq;
    }
    printf("[%s] GATE K=%d N=%d seed=0x%x inject=%d | repack_leaf vs oppX: mism=%d/%d maxrel=%.2e | vs ORACLE: mism=%d/%d maxrel=%.2e\n",
           tag,K,N,seed,(int)INJECT,m_opp,N,mr_opp,m_ora,N,mr_ora);
    if(has_opps) printf("[%s] OPPS-LAYOUT ggml_gemv_16x1 vs ORACLE: mism=%d/%d maxrel=%.2e => %s\n",
           tag,opps_mism,N,opps_maxrel,(opps_mism==0?"BYTE-COMPATIBLE (OPP-S timing VALID)":"LAYOUT-INCOMPATIBLE (OPP-S UNMEASURABLE here; needs ggml's own repack)"));
    int gate_ok = (m_ora==0);   // ZERO-MODEL oracle is the correctness authority
    if(!do_perf) return gate_ok?0:1;
    if(!gate_ok){ printf("[%s] PERF SKIPPED: oracle gate RED (0 fabricated numbers)\n",tag); return 1; }

    std::vector<float> oR(N),oO(N),oS(N); double tR[80],tO[80],tS[80];
    run_reps([&](){ ourfn((size_t)K,oR.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) oppfn(K,&oO[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1); },tO,reps);
    double rM=med(tR,reps),oM=med(tO,reps);
    printf("[%s] COLD K=%d N=%d reps=%d seed=0x%x | repack_leaf_ms=%.4f(iqr%.3f) oppX_ms=%.4f(iqr%.3f) | ratio_vs_oppX_CROSSOP=%.4f %s\n",
           tag,K,N,reps,seed,rM,reliqr(tR,reps),oM,reliqr(tO,reps),oM/rM,(oM/rM>=0.8?"PASS":"named-X"));
    if(has_opps&&oppsfn&&opps_mism==0){
        run_reps([&](){ oppsfn(K,oS.data(),0,(const void*)packed.data(),(const void*)act.data(),1,N); },tS,reps);
        double sM=med(tS,reps);
        printf("[%s] COLD-OPPS same-operator ggml_gemv_16x1_ms=%.4f(iqr%.3f) | ratio_vs_oppS=%.4f %s\n",
               tag,sM,reliqr(tS,reps),sM/rM,(sM/rM>=0.8?"PASS":"named-X"));
    }
    return 0;
}

// fillers
static void fW_q4_1(std::vector<block_q4_1>&W,int,std::mt19937&rng){ std::uniform_int_distribution<int>q(0,255);std::uniform_real_distribution<float>d(0.001f,0.05f),m(0.0f,0.01f);
    for(auto&b:W){b.d=(ggml_half)d(rng);b.m=(ggml_half)m(rng);for(int i=0;i<16;i++)b.qs[i]=(uint8_t)q(rng);} }
static void fW_q8_0(std::vector<block_q8_0>&W,int,std::mt19937&rng){ std::uniform_int_distribution<int>a(-127,127);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&b:W){b.d=(ggml_half)d(rng);for(int i=0;i<32;i++)b.qs[i]=(int8_t)a(rng);} }
static void fA_q8_0(std::vector<block_q8_0>&A,int,std::mt19937&rng){ std::uniform_int_distribution<int>a(-127,127);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&x:A){x.d=(ggml_half)d(rng);for(int i=0;i<32;i++)x.qs[i]=(int8_t)a(rng);} }
static void fA_q8_1(std::vector<block_q8_1>&A,int,std::mt19937&rng){ std::uniform_int_distribution<int>a(-127,127);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&x:A){x.d=(ggml_half)d(rng);x.s=(ggml_half)(d(rng)*10.f);for(int i=0;i<32;i++)x.qs[i]=(int8_t)a(rng);} }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s <q4_1|q8_0> K N reps [verify_only] [seed]\n",argv[0]); return 2; }
    const char* fmt=argv[1]; int K=atoi(argv[2]),N=atoi(argv[3]),reps=atoi(argv[4]);
    bool vo=argc>5&&atoi(argv[5])!=0; if(reps>80)reps=80; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    unsigned seed=argc>6?(unsigned)strtoul(argv[6],0,0):0xC0FFEE1u;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); if(!g_flush){fprintf(stderr,"OOM\n");return 3;} memset(g_flush,1,FLUSH_BYTES);
    int rc=0;
#ifdef FMT_Q4_1
    if(!strcmp(fmt,"q4_1"))      rc=run_fmt<block_q4_1,block_q8_1,block_q4_1x16>("q4_1",N,K,!vo,reps,seed,fW_q4_1,fA_q8_1,pack_q4_1,our_q4_1,ggml_vec_dot_q4_1_q8_1,ora_q4_1,1e-3,false,nullptr);
    else
#endif
#ifdef FMT_Q8_0
    if(!strcmp(fmt,"q8_0")) rc=run_fmt<block_q8_0,block_q8_0,block_q8_0x16>("q8_0",N,K,!vo,reps,seed,fW_q8_0,fA_q8_0,pack_q8_0,our_q8_0,ggml_vec_dot_q8_0_q8_0,ora_q8_0,1e-3,true,ggml_gemv_q8_0_16x1_q8_0);
    else
#endif
    { fprintf(stderr,"unknown/unbuilt fmt %s (this binary is built for one format only)\n",fmt); return 2; }
    free(g_flush); return rc;
}
