// A2-batch4 FLAT M=1 GEVM (decode) paired A/B driver.
//  OURS  = repack-GEVM LEAF (front-door lowerToRepackGemv product via weft-opt):
//          ONE GEVM call over nc columns, x16-interleaved weight + PLAIN single q8 act vector.
//  OPP   = the board's REAL dispatched block-dot ggml_vec_dot_<fmt> (stock libggml-cpu.so),
//          called per-column (nrc=1) over the SAME plain blocks == block-dot @ M=1.
//  GATE  = ZERO-MODEL independent scalar oracle (recomputed from plain inputs), per column.
//  ratio_cold = opp_med / ours_med   (>=0.8 => PASS ; <0.8 => named-X).
// cold: 32MiB flush (k1 64x L2 no L3 / rvv exceeds L2), CLOCK_MONOTONIC, median+relIQR, core-pinned.
// NOTE the leaf symbol string contains "vec_dot" but its BODY is typed_repack_gemv_loop_body
// (the repack-GEVM leaf) -- this is NOT the block-dot vec_dot path.
//  argv: <fmt q4_0|q5_0|q5_1> <K(mult32)> <N=nc(mult16)> <reps> [verify_only=0]
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
struct block_q4_0 { ggml_half d; uint8_t qs[16]; };                    // 18
struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; };      // 22
struct block_q5_1 { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[16]; }; // 24
struct block_q8_0 { ggml_half d; int8_t qs[32]; };                     // 34
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[32]; };        // 36
struct block_q4_0x16 { ggml_half d[16]; uint8_t qs[256]; };            // 288
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; };            // 352
struct block_q5_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; uint8_t qh[64]; }; // 384
static_assert(sizeof(block_q4_0)==18,"q4_0"); static_assert(sizeof(block_q8_0)==34,"q8_0");
static_assert(sizeof(block_q5_0)==22,"q5_0"); static_assert(sizeof(block_q5_1)==24,"q5_1");
static_assert(sizeof(block_q8_1)==36,"q8_1");
static_assert(sizeof(block_q4_0x16)==288,"q4_0x16");
static_assert(sizeof(block_q5_0x16)==352,"q5_0x16");
static_assert(sizeof(block_q5_1x16)==384,"q5_1x16");

// ---------------- OURS repack-GEVM leaf externs (board-selected ABI) ----------------
#ifdef BOARD_K1
extern "C" void weft_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
static inline void our_q4_0(size_t K, float* out, const uint8_t* packed, const uint8_t* act, size_t nc){
    weft_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(K, out, packed, act, nc); }
#else
extern "C" void weft_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv(
    size_t n, float* s, size_t nc, const uint8_t* vx, const uint8_t* vy);
static inline void our_q4_0(size_t K, float* out, const uint8_t* packed, const uint8_t* act, size_t nc){
    weft_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv(K, out, nc, packed, act); }
#endif
// q5_0 / q5_1: SAME 8-arg ABI on both boards (nc in arg3; bx/by/nrc unused).
extern "C" void weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
extern "C" void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
static inline void our_q5_0(size_t K,float*o,const uint8_t*p,const uint8_t*a,size_t nc){ weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(K,o,nc,p,0,a,0,0);}
static inline void our_q5_1(size_t K,float*o,const uint8_t*p,const uint8_t*a,size_t nc){ weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(K,o,nc,p,0,a,0,0);}

// ---------------- OPP stock-.so block-dot ----------------
extern "C" void ggml_vec_dot_q4_0_q8_0(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);

// ---------------- x16 repack (byte-exact to leaf addressing; mirrors flat_gemm_paired_driver) ----------------
static block_q4_0x16 pack_q4_0(block_q4_0* in){ block_q4_0x16 o; for(int c=0;c<16;c++)o.d[c]=in[c].d;
    for(int b=0;b<16;b++)for(int c=0;c<16;c++)o.qs[b*16+c]=(uint8_t)(in[c].qs[b]^0x88); return o; }
static block_q5_0x16 pack_q5_0(block_q5_0* in){ block_q5_0x16 o; for(int c=0;c<16;c++)o.d[c]=in[c].d;
    for(int i=0;i<256;++i){int id=i%16,of=i/16;o.qs[i]=in[id].qs[of];}
    uint32_t qc[16]; for(int c=0;c<16;c++)std::memcpy(&qc[c],in[c].qh,4);
    for(int k=0;k<16;k++){uint16_t lo=0,hi=0;for(int c=0;c<16;c++){lo|=(uint16_t)(((qc[c]>>k)&1u)<<c);hi|=(uint16_t)(((qc[c]>>(k+16))&1u)<<c);}std::memcpy(o.qh+k*2,&lo,2);std::memcpy(o.qh+32+k*2,&hi,2);} return o; }
static block_q5_1x16 pack_q5_1(block_q5_1* in){ block_q5_1x16 o; for(int c=0;c<16;c++){o.d[c]=in[c].d;o.m[c]=in[c].m;}
    for(int i=0;i<256;++i){int id=i%16,of=i/16;o.qs[i]=in[id].qs[of];}
    uint32_t qc[16]; for(int c=0;c<16;c++)std::memcpy(&qc[c],in[c].qh,4);
    for(int k=0;k<16;k++){uint16_t lo=0,hi=0;for(int c=0;c<16;c++){lo|=(uint16_t)(((qc[c]>>k)&1u)<<c);hi|=(uint16_t)(((qc[c]>>(k+16))&1u)<<c);}std::memcpy(o.qh+k*2,&lo,2);std::memcpy(o.qh+32+k*2,&hi,2);} return o; }

// ---------------- ZERO-MODEL scalar oracles (from PLAIN blocks) ----------------
static float ora_q4_0(const block_q4_0&w,const block_q8_0&a){ int s=0; for(int j=0;j<16;j++){int x0=(w.qs[j]&0xF)-8,x1=(w.qs[j]>>4)-8; s+=x0*a.qs[j]+x1*a.qs[j+16];} return (float)s*(float)w.d*(float)a.d; }
static float ora_q5_0(const block_q5_0&w,const block_q8_0&a){ uint32_t qh;std::memcpy(&qh,w.qh,4);int s=0;
    for(int j=0;j<16;j++){uint8_t h0=(uint8_t)(((qh>>j)&1)<<4),h1=(uint8_t)(((qh>>(j+16))&1)<<4);
        int x0=(int8_t)((w.qs[j]&0xF)|h0)-16,x1=(int8_t)((w.qs[j]>>4)|h1)-16; s+=x0*a.qs[j]+x1*a.qs[j+16];}
    return (float)s*(float)w.d*(float)a.d; }
static float ora_q5_1(const block_q5_1&w,const block_q8_1&a){ uint32_t qh;std::memcpy(&qh,w.qh,4);int s=0;
    for(int j=0;j<16;j++){uint8_t h0=(uint8_t)(((qh>>j)&1)<<4),h1=(uint8_t)(((qh>>(j+16))&1)<<4);
        int x0=(w.qs[j]&0xF)|h0,x1=(w.qs[j]>>4)|h1; s+=x0*a.qs[j]+x1*a.qs[j+16];}
    return (float)w.d*(float)a.d*(float)s+(float)w.m*(float)a.s; }

// ---------------- cold timing infra ----------------
static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static const size_t FLUSH_BYTES=32ull*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&&f,double*out,int reps){ for(int r=0;r<reps;r++){ cold_flush(); double a=now_ms(); f(); out[r]=now_ms()-a; } }
static double med(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

// ---------------- per-format test (nc=N cols, single M=1 act vector) ----------------
template<class BW,class BA,class BP>
static int run_fmt(const char* tag,int N,int K,bool do_perf,int reps,unsigned seed,
                   void(*fillW)(std::vector<BW>&,int,std::mt19937&),
                   void(*fillA)(std::vector<BA>&,int,std::mt19937&),
                   BP(*pack)(BW*),
                   void(*ourfn)(size_t,float*,const uint8_t*,const uint8_t*,size_t),
                   void(*oppfn)(int,float*,size_t,const void*,size_t,const void*,size_t,int),
                   float(*orafn)(const BW&,const BA&), double reltol){
    int nb=K/32; std::mt19937 rng(seed);
    std::vector<BW> W((size_t)N*nb); fillW(W,nb,rng);
    std::vector<BA> act(nb); fillA(act,nb,rng);
    int ng=N/16; std::vector<BP> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++)for(int b=0;b<nb;b++){ BW tmp[16]; for(int c=0;c<16;c++)tmp[c]=W[(size_t)(g*16+c)*nb+b]; packed[(size_t)g*nb+b]=pack(tmp); }

    std::vector<float> ours(N,0.f), ropp(N,0.f), rora(N,0.f);
    ourfn((size_t)K,ours.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N);
    for(int r=0;r<N;r++){ oppfn(K,&ropp[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1);
        double s=0; for(int b=0;b<nb;b++)s+=orafn(W[(size_t)r*nb+b],act[b]); rora[r]=(float)s; }
    int m_opp=0,m_ora=0; float mr_opp=0,mr_ora=0;
    for(int r=0;r<N;r++){
        float ao=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f);
        float aq=std::fabs(ours[r]-rora[r])/(std::fabs(rora[r])+1e-6f);
        if(ao>=reltol)m_opp++; if(aq>=reltol)m_ora++;
        if(ao>mr_opp)mr_opp=ao; if(aq>mr_ora)mr_ora=aq;
    }
    printf("[%s] GATE K=%d N=%d seed=0x%x | repack_leaf vs opp: mism=%d/%d maxrel=%.2e | vs oracle: mism=%d/%d maxrel=%.2e\n",
           tag,K,N,seed,m_opp,N,mr_opp,m_ora,N,mr_ora);
    int gate_ok = (m_ora==0);   // ZERO-MODEL oracle is the correctness authority
    if(!do_perf) return gate_ok?0:1;

    std::vector<float> oR(N),oO(N); double tR[80],tO[80];
    run_reps([&](){ ourfn((size_t)K,oR.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) oppfn(K,&oO[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1); },tO,reps);
    double rM=med(tR,reps),oM=med(tO,reps);
    printf("[%s] COLD K=%d N=%d reps=%d seed=0x%x | repack_leaf_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f %s\n",
           tag,K,N,reps,seed,rM,reliqr(tR,reps),oM,reliqr(tO,reps),oM/rM,(oM/rM>=0.8?"PASS":"named-X"));
    return gate_ok?0:1;
}

// fillers
static void fW_q4_0(std::vector<block_q4_0>&W,int,std::mt19937&rng){ std::uniform_int_distribution<int>q(0,255);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&b:W){b.d=(ggml_half)d(rng);for(int i=0;i<16;i++)b.qs[i]=(uint8_t)q(rng);} }
static void fA_q8_0(std::vector<block_q8_0>&A,int,std::mt19937&rng){ std::uniform_int_distribution<int>a(-127,127);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&x:A){x.d=(ggml_half)d(rng);for(int i=0;i<32;i++)x.qs[i]=(int8_t)a(rng);} }
static void fW_q5_0(std::vector<block_q5_0>&W,int,std::mt19937&rng){ std::uniform_int_distribution<int>q(0,255);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&b:W){b.d=(ggml_half)d(rng);for(int i=0;i<4;i++)b.qh[i]=(uint8_t)q(rng);for(int i=0;i<16;i++)b.qs[i]=(uint8_t)q(rng);} }
static void fW_q5_1(std::vector<block_q5_1>&W,int,std::mt19937&rng){ std::uniform_int_distribution<int>q(0,255);std::uniform_real_distribution<float>d(0.001f,0.05f),m(0.0f,0.01f);
    for(auto&b:W){b.d=(ggml_half)d(rng);b.m=(ggml_half)m(rng);for(int i=0;i<4;i++)b.qh[i]=(uint8_t)q(rng);for(int i=0;i<16;i++)b.qs[i]=(uint8_t)q(rng);} }
static void fA_q8_1(std::vector<block_q8_1>&A,int,std::mt19937&rng){ std::uniform_int_distribution<int>a(-127,127);std::uniform_real_distribution<float>d(0.001f,0.05f);
    for(auto&x:A){x.d=(ggml_half)d(rng);x.s=(ggml_half)(d(rng)*10.f);for(int i=0;i<32;i++)x.qs[i]=(int8_t)a(rng);} }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s <q4_0|q5_0|q5_1> K N reps [verify_only]\n",argv[0]); return 2; }
    const char* fmt=argv[1]; int K=atoi(argv[2]),N=atoi(argv[3]),reps=atoi(argv[4]);
    bool vo=argc>5&&atoi(argv[5])!=0; if(reps>80)reps=80; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    unsigned seed=argc>6?(unsigned)strtoul(argv[6],0,0):0xC0FFEE1u;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); if(!g_flush){fprintf(stderr,"OOM\n");return 3;} memset(g_flush,1,FLUSH_BYTES);
    int rc=0;
    if(!strcmp(fmt,"q4_0")) rc=run_fmt<block_q4_0,block_q8_0,block_q4_0x16>("q4_0",N,K,!vo,reps,seed,fW_q4_0,fA_q8_0,pack_q4_0,our_q4_0,ggml_vec_dot_q4_0_q8_0,ora_q4_0,1e-3);
    else if(!strcmp(fmt,"q5_0")) rc=run_fmt<block_q5_0,block_q8_0,block_q5_0x16>("q5_0",N,K,!vo,reps,seed,fW_q5_0,fA_q8_0,pack_q5_0,our_q5_0,ggml_vec_dot_q5_0_q8_0,ora_q5_0,1e-3);
    else if(!strcmp(fmt,"q5_1")) rc=run_fmt<block_q5_1,block_q8_1,block_q5_1x16>("q5_1",N,K,!vo,reps,seed,fW_q5_1,fA_q8_1,pack_q5_1,our_q5_1,ggml_vec_dot_q5_1_q8_1,ora_q5_1,1e-3);
    else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }
    free(g_flush); return rc;
}
