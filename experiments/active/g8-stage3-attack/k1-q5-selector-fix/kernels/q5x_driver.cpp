// G8 §六.3 q5_0/q5_1 k1(VLEN256) deployed-core verification + repack what-if driver.
// Purpose:
//  (1) byte-exact BOTH the deployed BLOCK-DOT vec_dot kernel AND the (non-deployed,
//      force-constructed) REDESIGN-B repack-mf2 GEVM leaf against the REAL stock-.so
//      opponent ggml_vec_dot_q5_{0,1}_q8_{0,1} AND an independent scalar oracle.
//  (2) cold M=1 kernel-sym timing, 3 paired regions vs the SAME opponent:
//        - blockdot_ours : N per-column vec_dot calls (== the DEPLOYED decode core)
//        - repack_leaf   : ONE GEVM call over N columns (the non-deployed repack path)
//        - opp           : N per-column ggml_vec_dot calls (stock .so, native-RVV)
//  ratio_cold = opp_med / ours_med  (>=0.8 => PASS gate). k1: 32MiB flush (64x L2, no L3),
//  CLOCK_MONOTONIC (no cache-miss PMU), median + relIQR, core-pinned by launcher.
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

// ---------------- q5_0 ----------------
#define QK5_0 32
#define QK8_0 32
struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; };   // 22B
struct block_q8_0 { ggml_half d; int8_t  qs[QK8_0];   };                  // 34B
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; }; // 352B
static_assert(sizeof(block_q5_0)==22,"q5_0");
static_assert(sizeof(block_q8_0)==34,"q8_0");
static_assert(sizeof(block_q5_0x16)==352,"q5_0x16");

static block_q5_0x16 make_block_q5_0x16(block_q5_0 * in) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    for (int i = 0; i < 256; ++i) { int src_id=i%16, src_off=i/16; out.qs[i]=in[src_id].qs[src_off]; }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    for (int k = 0; k < 16; k++) {
        uint16_t lo=0, hi=0;
        for (int c = 0; c < 16; c++) { lo|=(uint16_t)(((qh_col[c]>>k)&1u)<<c); hi|=(uint16_t)(((qh_col[c]>>(k+16))&1u)<<c); }
        std::memcpy(out.qh+0+k*2,&lo,2); std::memcpy(out.qh+32+k*2,&hi,2);
    }
    return out;
}
static float oracle_q5_0(const block_q5_0 & w, const block_q8_0 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int isum = 0;
    for (int j = 0; j < QK5_0/2; j++) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12)) >> 0) & 0x10;
        const int w0 = ((w.qs[j] & 0x0F) | xh_0) - 16;
        const int w1 = ((w.qs[j] >>   4) | xh_1) - 16;
        isum += w0 * a.qs[j] + w1 * a.qs[j + 16];
    }
    return (float)isum * (float)w.d * (float)a.d;
}

// ---------------- q5_1 ----------------
#define QK5_1 32
#define QK8_1 32
struct block_q5_1 { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[QK5_1/2]; }; // 24B
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK8_1]; };                    // 36B
struct block_q5_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; uint8_t qh[64]; }; // 384B
static_assert(sizeof(block_q5_1)==24,"q5_1");
static_assert(sizeof(block_q8_1)==36,"q8_1");
static_assert(sizeof(block_q5_1x16)==384,"q5_1x16");

static block_q5_1x16 make_block_q5_1x16(block_q5_1 * in) {
    block_q5_1x16 out;
    for (int i = 0; i < 16; i++) { out.d[i]=in[i].d; out.m[i]=in[i].m; }
    for (int i = 0; i < 256; ++i) { int src_id=i%16, src_off=i/16; out.qs[i]=in[src_id].qs[src_off]; }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    for (int k = 0; k < 16; k++) {
        uint16_t lo=0, hi=0;
        for (int c = 0; c < 16; c++) { lo|=(uint16_t)(((qh_col[c]>>k)&1u)<<c); hi|=(uint16_t)(((qh_col[c]>>(k+16))&1u)<<c); }
        std::memcpy(out.qh+0+k*2,&lo,2); std::memcpy(out.qh+32+k*2,&hi,2);
    }
    return out;
}
static float oracle_q5_1(const block_q5_1 & w, const block_q8_1 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int sumi = 0;
    for (int j = 0; j < QK5_1/2; j++) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12)) >> 0) & 0x10;
        const int32_t x0 = (w.qs[j] & 0xF) | xh_0;
        const int32_t x1 = (w.qs[j] >>  4) | xh_1;
        sumi += (x0 * a.qs[j]) + (x1 * a.qs[j + 16]);
    }
    return (float)w.d * (float)a.d * (float)sumi + (float)w.m * (float)a.s;
}

// ---------- ours kernels ----------
// repack-mf2 GEVM leaf (force-constructed VLEN256; front-door DECLINES it at decode)
extern "C" void weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
extern "C" void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc);
// deployed BLOCK-DOT vec_dot (per-column) -- the REAL decode dispatch on k1 VLEN256
extern "C" void weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot(
    size_t n, float* out, const uint8_t* x, const uint8_t* y, const int32_t* nrc);
extern "C" void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot(
    size_t n, float* out, const uint8_t* x, const uint8_t* y, const int32_t* nrc);

// ---------- real stock-.so opponent ----------
extern "C" void ggml_vec_dot_q5_0_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }
static const size_t FLUSH_BYTES = 32ull*1024*1024;  // 64x k1 L2 (512KiB), no L3
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&& f, double* out, int reps){ for(int r=0;r<reps;r++){ cold_flush(); double a=now_ms(); f(); out[r]=now_ms()-a; } }
static double med(double* v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double* v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

static int test_q5_0(int N, int K, bool do_perf, int reps){
    int nb=K/32; std::mt19937 rng(0xC0FFEE1);
    std::uniform_int_distribution<int> qd(0,255), qhd(0,255); std::uniform_real_distribution<float> dd(0.001f,0.05f);
    std::vector<block_q5_0> W((size_t)N*nb);
    for(auto&b:W){ b.d=(ggml_half)dd(rng); for(int i=0;i<4;i++)b.qh[i]=(uint8_t)qhd(rng); for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng); }
    std::vector<block_q8_0> act(nb); std::uniform_int_distribution<int> ad(-127,127);
    for(auto&a:act){ a.d=(ggml_half)dd(rng); for(int i=0;i<32;i++)a.qs[i]=(int8_t)ad(rng); }
    int ng=N/16; std::vector<block_q5_0x16> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++)for(int blk=0;blk<nb;blk++){ block_q5_0 tmp[16]; for(int c=0;c<16;c++)tmp[c]=W[(size_t)(g*16+c)*nb+blk]; packed[(size_t)g*nb+blk]=make_block_q5_0x16(tmp); }

    std::vector<float> ours(N,0.f), oursBD(N,0.f), ropp(N,0.f), rora(N,0.f); int32_t nrc1=1;
    weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0((size_t)K, ours.data(),(size_t)N,(const uint8_t*)packed.data(),0,(const uint8_t*)act.data(),0,0);
    for(int r=0;r<N;r++){
        weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot((size_t)K,&oursBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1);
        ggml_vec_dot_q5_0_q8_0(K,&ropp[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1);
        double s=0; for(int b=0;b<nb;b++)s+=oracle_q5_0(W[(size_t)r*nb+b],act[b]); rora[r]=(float)s;
    }
    int m_rep_opp=0,m_bd_opp=0,m_ora=0; float mr_rep=0,mr_bd=0;
    for(int r=0;r<N;r++){
        if(std::memcmp(&ours[r],&ropp[r],4)!=0)m_rep_opp++;
        if(std::memcmp(&oursBD[r],&ropp[r],4)!=0)m_bd_opp++;
        if(std::fabs(ours[r]-rora[r])/(std::fabs(rora[r])+1e-6f)>=1e-3f)m_ora++;
        float a=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(a>mr_rep)mr_rep=a;
        float b=std::fabs(oursBD[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(b>mr_bd)mr_bd=b;
    }
    printf("[q5_0] BYTEEXACT repack_leaf vs opp: mism(bitf32)=%d/%d max_rel=%.2e | blockdot vs opp: mism=%d/%d max_rel=%.2e | vs oracle(rel>1e-3): %d/%d\n",
           m_rep_opp,N,mr_rep,m_bd_opp,N,mr_bd,m_ora,N);
    if(!do_perf) return (m_rep_opp==0&&m_bd_opp==0&&m_ora==0)?0:1;

    std::vector<float> oR(N),oBD(N),oO(N); double tR[64],tBD[64],tO[64];
    run_reps([&](){ weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0((size_t)K,oR.data(),(size_t)N,(const uint8_t*)packed.data(),0,(const uint8_t*)act.data(),0,0); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot((size_t)K,&oBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1); },tBD,reps);
    run_reps([&](){ for(int r=0;r<N;r++) ggml_vec_dot_q5_0_q8_0(K,&oO[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1); },tO,reps);
    double rM=med(tR,reps),bM=med(tBD,reps),oM=med(tO,reps);
    printf("[q5_0] COLD K=%d N=%d reps=%d | repack_leaf_ms=%.4f(iqr%.3f) blockdot_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f ratio_blockdot=%.4f\n",
           K,N,reps,rM,reliqr(tR,reps),bM,reliqr(tBD,reps),oM,reliqr(tO,reps),oM/rM,oM/bM);
    return 0;
}

static int test_q5_1(int N, int K, bool do_perf, int reps){
    int nb=K/32; std::mt19937 rng(0xC0FFEE2);
    std::uniform_int_distribution<int> qd(0,255), qhd(0,255); std::uniform_real_distribution<float> dd(0.001f,0.05f), md_(0.0f,0.01f);
    std::vector<block_q5_1> W((size_t)N*nb);
    for(auto&b:W){ b.d=(ggml_half)dd(rng); b.m=(ggml_half)md_(rng); for(int i=0;i<4;i++)b.qh[i]=(uint8_t)qhd(rng); for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng); }
    std::vector<block_q8_1> act(nb); std::uniform_int_distribution<int> ad(-127,127);
    for(auto&a:act){ a.d=(ggml_half)dd(rng); a.s=(ggml_half)(dd(rng)*10.f); for(int i=0;i<32;i++)a.qs[i]=(int8_t)ad(rng); }
    int ng=N/16; std::vector<block_q5_1x16> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++)for(int blk=0;blk<nb;blk++){ block_q5_1 tmp[16]; for(int c=0;c<16;c++)tmp[c]=W[(size_t)(g*16+c)*nb+blk]; packed[(size_t)g*nb+blk]=make_block_q5_1x16(tmp); }

    std::vector<float> ours(N,0.f), oursBD(N,0.f), ropp(N,0.f), rora(N,0.f); int32_t nrc1=1;
    weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1((size_t)K, ours.data(),(size_t)N,(const uint8_t*)packed.data(),0,(const uint8_t*)act.data(),0,0);
    for(int r=0;r<N;r++){
        weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot((size_t)K,&oursBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1);
        ggml_vec_dot_q5_1_q8_1(K,&ropp[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1);
        double s=0; for(int b=0;b<nb;b++)s+=oracle_q5_1(W[(size_t)r*nb+b],act[b]); rora[r]=(float)s;
    }
    // q5_1 random-scale: use rel>1e-3 tolerance (scalar-vs-vector FMA-order benign noise; see rvv casefile)
    int m_rep=0,m_bd=0,m_ora=0; float mr_rep=0,mr_bd=0;
    for(int r=0;r<N;r++){
        if(std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f)>=1e-3f)m_rep++;
        if(std::fabs(oursBD[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f)>=1e-3f)m_bd++;
        if(std::fabs(ours[r]-rora[r])/(std::fabs(rora[r])+1e-6f)>=1e-3f)m_ora++;
        float a=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(a>mr_rep)mr_rep=a;
        float b=std::fabs(oursBD[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(b>mr_bd)mr_bd=b;
    }
    printf("[q5_1] rel>1e-3 repack_leaf vs opp: mism=%d/%d max_rel=%.2e | blockdot vs opp: mism=%d/%d max_rel=%.2e | vs oracle: %d/%d\n",
           m_rep,N,mr_rep,m_bd,N,mr_bd,m_ora,N);
    if(!do_perf) return (m_rep==0&&m_bd==0&&m_ora==0)?0:1;

    std::vector<float> oR(N),oBD(N),oO(N); double tR[64],tBD[64],tO[64];
    run_reps([&](){ weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1((size_t)K,oR.data(),(size_t)N,(const uint8_t*)packed.data(),0,(const uint8_t*)act.data(),0,0); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot((size_t)K,&oBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1); },tBD,reps);
    run_reps([&](){ for(int r=0;r<N;r++) ggml_vec_dot_q5_1_q8_1(K,&oO[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1); },tO,reps);
    double rM=med(tR,reps),bM=med(tBD,reps),oM=med(tO,reps);
    printf("[q5_1] COLD K=%d N=%d reps=%d | repack_leaf_ms=%.4f(iqr%.3f) blockdot_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f ratio_blockdot=%.4f\n",
           K,N,reps,rM,reliqr(tR,reps),bM,reliqr(tBD,reps),oM,reliqr(tO,reps),oM/rM,oM/bM);
    return 0;
}

int main(int argc, char** argv){
    // argv: <fmt q5_0|q5_1> <K> <N> <reps> [verify_only=0]
    if(argc<5){ fprintf(stderr,"usage: %s <q5_0|q5_1> K N reps [verify_only]\n",argv[0]); return 2; }
    const char* fmt=argv[1]; int K=atoi(argv[2]), N=atoi(argv[3]), reps=atoi(argv[4]);
    bool verify_only = argc>5 && atoi(argv[5])!=0;
    if(reps>64)reps=64; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); if(!g_flush){ fprintf(stderr,"OOM\n"); return 3; } memset(g_flush,1,FLUSH_BYTES);
    int rc=0;
    if(!strcmp(fmt,"q5_0")) rc=test_q5_0(N,K,!verify_only,reps);
    else if(!strcmp(fmt,"q5_1")) rc=test_q5_1(N,K,!verify_only,reps);
    else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }
    free(g_flush); return rc;
}
