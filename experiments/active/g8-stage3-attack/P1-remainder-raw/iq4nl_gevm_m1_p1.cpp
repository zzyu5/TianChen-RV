// iq4nl_gevm_m1_p1.cpp — P1 #4 (rvv) / #6 (k1): iq4_nl gemm_tile DECODE, independent M=1 GEVM.
// Derived from k1-gevm-sweep/raw/iq4nl_driver.cpp (M=1 protocol 100% reused) per P1 prereg §1.2B.
// Added per prereg: (1) seed argv (upstream had rng(0x14E00D) hardcoded => 2-seed impossible),
//                   (2) OPP-S same-operator opponent ggml_gemv_iq4_nl_16x1_q8_0 (成色权威),
//                   (3) per-rep dump for bootstrap CI, (4) 3-way A/B/C interleave within-proc.
//
// M=1 DECODE (prereg §4.2, 禁继承): activation = ONE PLAIN block_q8_0 vector (stride 34, no row
// interleave); weight = x16 interleaved (block_iq4_nlx16 288B); ONE GEVM call over nc columns.
// NOT inherited from prefill nr16, NOT from vec_dot M=1, NOT from any historical anchor.
//
// OPP-X (CROSSOP) = ggml_vec_dot_iq4_nl_q8_0 per column (block-dot, nrc=1)  -> fills T3 cell
// OPP-S (same-op)  = ggml_gemv_iq4_nl_16x1_q8_0 (hand-written RVV, arch/riscv/repack.cpp) -> 成色权威
// ZERO-MODEL: every column recomputed from PLAIN inputs (oracle_iq4_nl), zero reuse of intermediates.
// OPP-S layout gate: OPP-S consumes block_iq4_nlx16; if make_x16 disagreed with ggml's expected
// layout the ZERO-MODEL check on OPP-S output would fail => functional layout gate (reported).
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
#include <riscv_vector.h>

typedef _Float16 ggml_half;
#define QK4_NL 32
static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

struct block_iq4_nl   { ggml_half d; uint8_t qs[QK4_NL/2]; };  // 18B
struct block_q8_0     { ggml_half d; int8_t  qs[QK4_NL];   };  // 34B
struct block_iq4_nlx16{ ggml_half d[16]; uint8_t qs[256];  };  // 288B
static_assert(sizeof(block_iq4_nl)==18,"iq4_nl");
static_assert(sizeof(block_q8_0)==34,"q8_0");
static_assert(sizeof(block_iq4_nlx16)==288,"iq4_nlx16");

static block_iq4_nlx16 make_x16(block_iq4_nl* in){
    block_iq4_nlx16 o;
    for(int i=0;i<16;i++) o.d[i]=in[i].d;
    for(int i=0;i<256;++i){ int src=i%16, off=i/16; o.qs[i]=in[src].qs[off]; }
    return o;
}
static float oracle_iq4_nl(const block_iq4_nl& w, const block_q8_0& a){
    int isum=0;
    for(int j=0;j<QK4_NL/2;j++){
        int w0=kvalues_iq4nl[w.qs[j]&0xF];
        int w1=kvalues_iq4nl[w.qs[j]>>4];
        isum += w0*a.qs[j] + w1*a.qs[j+QK4_NL/2];
    }
    return (float)isum * (float)w.d * (float)a.d;
}

// OURS: front-door lowered repack GEVM leaf (n, s, vx, vy, nc)
extern "C" void weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
// OPP-X: as-shipped block-dot (VLEN dispatch thunk -> _vl128/_vl256)
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
// OPP-S: as-shipped same-operator hand-written RVV repack-GEVM
extern "C" void ggml_gemv_iq4_nl_16x1_q8_0(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);

#ifndef FLUSH_MB
#define FLUSH_MB 224
#endif
static const size_t FLUSH=(size_t)FLUSH_MB*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double riqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
    double m=v[n/2]; return m>0?100.0*(v[3*n/4]-v[n/4])/m:0.0; }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s K nc reps seed [verify_only]\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nc=atoi(argv[2]), reps=atoi(argv[3]);
    unsigned seed=(unsigned)strtoul(argv[4],0,0);
    bool vo = argc>5 && atoi(argv[5]);
    if(nc%16){ fprintf(stderr,"nc must be mult of 16\n"); return 2; }
    if(K%QK4_NL){ fprintf(stderr,"K must be mult of 32\n"); return 2; }
    if(reps<10) reps=10;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH); if(!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    memset(g_flush,1,FLUSH);
    int nb=K/QK4_NL;
    long vlen=(long)__riscv_vlenb()*8;

    std::mt19937 rng(seed);                       // <-- P1: seed is now a parameter (2-seed enabled)
    std::uniform_int_distribution<int> qd(0,255);
    std::uniform_real_distribution<float> dd(0.001f,0.05f);
    std::uniform_int_distribution<int> ad(-127,127);

    // PLAIN weights (opponent-X native form) — the single source of truth for ZERO-MODEL
    std::vector<block_iq4_nl> W((size_t)nc*nb);
    for(auto&b:W){ b.d=(ggml_half)dd(rng); for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng); }
    // M=1 activation: ONE PLAIN q8_0 vector (stride 34, NO row interleave)
    std::vector<block_q8_0> act(nb);
    for(auto&a:act){ a.d=(ggml_half)dd(rng); for(int i=0;i<32;i++)a.qs[i]=(int8_t)ad(rng); }
    // x16 interleaved weights (ours + OPP-S consume this)
    int ng=nc/16; std::vector<block_iq4_nlx16> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++) for(int l=0;l<nb;l++){
        block_iq4_nl tmp[16]; for(int c=0;c<16;c++) tmp[c]=W[(size_t)(g*16+c)*nb+l];
        packed[(size_t)g*nb+l]=make_x16(tmp);
    }

    std::vector<float> ours(nc,0.f), oX(nc,0.f), oS(nc,0.f), ora(nc,0.f);
    weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
        (size_t)K, ours.data(), (const uint8_t*)packed.data(), (const uint8_t*)act.data(), (size_t)nc);
    for(int c=0;c<nc;c++) ggml_vec_dot_iq4_nl_q8_0(K,&oX[c],0,W.data()+(size_t)c*nb,0,act.data(),0,1);
    ggml_gemv_iq4_nl_16x1_q8_0(K, oS.data(), (size_t)nc, (const void*)packed.data(), (const void*)act.data(), 1, nc);
    for(int c=0;c<nc;c++){ double s=0; for(int l=0;l<nb;l++) s+=oracle_iq4_nl(W[(size_t)c*nb+l],act[l]); ora[c]=(float)s; }

    int m_ours=0,m_X=0,m_S=0; float mr_ours=0,mr_X=0,mr_S=0;
    for(int c=0;c<nc;c++){
        float den=std::fabs(ora[c])+1e-6f;
        float a=std::fabs(ours[c]-ora[c])/den, b=std::fabs(oX[c]-ora[c])/den, d=std::fabs(oS[c]-ora[c])/den;
        if(a>=1e-3f)m_ours++; if(b>=1e-3f)m_X++; if(d>=1e-3f)m_S++;
        if(a>mr_ours)mr_ours=a; if(b>mr_X)mr_X=b; if(d>mr_S)mr_S=d;
    }
    printf("ZERO-MODEL K=%d nc=%d VLEN=%ld seed=0x%X | ours vs oracle: mism=%d/%d max_rel=%.2e | "
           "OPP-X vs oracle: mism=%d/%d max_rel=%.2e | OPP-S vs oracle: mism=%d/%d max_rel=%.2e => %s | "
           "OPP-S layout gate: %s\n",
           K,nc,vlen,seed,m_ours,nc,mr_ours,m_X,nc,mr_X,m_S,nc,mr_S,
           (m_ours==0&&m_X==0&&m_S==0)?"ALL-OK":"*** MISMATCH ***",
           (m_S==0)?"PASS(make_x16 == ggml block_iq4_nlx16 expectation, functionally proven)":"FAIL=>VOID-S");
    if(vo){ free(g_flush); return (m_ours==0&&m_X==0&&m_S==0)?0:1; }
    if(m_ours!=0){ fprintf(stderr,"VOID-CORRECTNESS: ours mismatch\n"); free(g_flush); return 1; }

    // COLD paired A/B/C, same-session interleave, flush before EACH timed region
    std::vector<double> tR,tX,tS; tR.reserve(reps); tX.reserve(reps); tS.reserve(reps);
    std::vector<float> bR(nc),bX(nc),bS(nc);
    cold_flush(); weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0((size_t)K,bR.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)nc);
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns();
        weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0((size_t)K,bR.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)nc);
        tR.push_back(now_ns()-t0);
        cold_flush(); double t1=now_ns();
        for(int c=0;c<nc;c++) ggml_vec_dot_iq4_nl_q8_0(K,&bX[c],0,W.data()+(size_t)c*nb,0,act.data(),0,1);
        tX.push_back(now_ns()-t1);
        cold_flush(); double t2=now_ns();
        ggml_gemv_iq4_nl_16x1_q8_0(K,bS.data(),(size_t)nc,(const void*)packed.data(),(const void*)act.data(),1,nc);
        tS.push_back(now_ns()-t2);
    }
    double rM=med(tR), xM=med(tX), sM=med(tS);
    printf("REPS_OURS seed=0x%X :", seed); for(double v:tR) printf(" %.0f", v); printf("\n");
    printf("REPS_OPPX seed=0x%X :", seed); for(double v:tX) printf(" %.0f", v); printf("\n");
    printf("REPS_OPPS seed=0x%X :", seed); for(double v:tS) printf(" %.0f", v); printf("\n");
    printf("GEVM_M1 K=%d nc=%d reps=%d VLEN=%ld seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "oppX_med_ns=%.0f(iqr%.2f%%) oppS_med_ns=%.0f(iqr%.2f%%) | ratio_cold_X=%.4f ratio_cold_S=%.4f\n",
           K,nc,reps,vlen,seed,rM,riqr(tR),xM,riqr(tX),sM,riqr(tS),xM/rM,sM/rM);
    free(g_flush); return 0;
}
