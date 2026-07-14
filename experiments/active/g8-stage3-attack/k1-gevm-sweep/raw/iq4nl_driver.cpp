// G8 §六.3 group3: iq4_nl @k1(VLEN256) deployed block-dot + repack what-if.
// repack decode is DECLINED by the front door at VLEN256 (same gate as q5); this force-
// constructs the repack-mf2 GEVM leaf and measures it (+ the deployed block-dot) vs the
// stock-.so opponent ggml_vec_dot_iq4_nl_q8_0. byte-exact vs opponent AND independent oracle.
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
#define QK4_NL 32
static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

struct block_iq4_nl { ggml_half d; uint8_t qs[QK4_NL/2]; };   // 18B
struct block_q8_0    { ggml_half d; int8_t qs[QK4_NL]; };      // 34B
struct block_iq4_nlx16 { ggml_half d[16]; uint8_t qs[256]; };  // 288B
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

extern "C" void weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
extern "C" void weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(
    size_t n, float* out, const uint8_t* x, const uint8_t* y, const int32_t* nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static const size_t FLUSH=32ull*1024*1024; static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&&f,double*o,int r){ for(int i=0;i<r;i++){ cold_flush(); double a=now_ms(); f(); o[i]=now_ms()-a; } }
static double med(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double riqr(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[3*n/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s iq4_nl K N reps [verify]\n",argv[0]); return 2; }
    int K=atoi(argv[2]), N=atoi(argv[3]), reps=atoi(argv[4]); bool vo=argc>5&&atoi(argv[5]);
    if(reps>64)reps=64; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH); if(!g_flush){ fprintf(stderr,"OOM\n"); return 3; } memset(g_flush,1,FLUSH);
    int nb=K/QK4_NL;
    std::mt19937 rng(0x14E00D); std::uniform_int_distribution<int> qd(0,255); std::uniform_real_distribution<float> dd(0.001f,0.05f);
    std::vector<block_iq4_nl> W((size_t)N*nb);
    for(auto&b:W){ b.d=(ggml_half)dd(rng); for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng); }
    std::vector<block_q8_0> act(nb); std::uniform_int_distribution<int> ad(-127,127);
    for(auto&a:act){ a.d=(ggml_half)dd(rng); for(int i=0;i<32;i++)a.qs[i]=(int8_t)ad(rng); }
    int ng=N/16; std::vector<block_iq4_nlx16> packed((size_t)ng*nb);
    for(int g=0;g<ng;g++)for(int blk=0;blk<nb;blk++){ block_iq4_nl tmp[16]; for(int c=0;c<16;c++)tmp[c]=W[(size_t)(g*16+c)*nb+blk]; packed[(size_t)g*nb+blk]=make_x16(tmp); }

    std::vector<float> ours(N,0.f), oursBD(N,0.f), ropp(N,0.f), rora(N,0.f); int32_t nrc1=1;
    weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0((size_t)K, ours.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N);
    for(int r=0;r<N;r++){
        weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot((size_t)K,&oursBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1);
        ggml_vec_dot_iq4_nl_q8_0(K,&ropp[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1);
        double s=0; for(int b=0;b<nb;b++)s+=oracle_iq4_nl(W[(size_t)r*nb+b],act[b]); rora[r]=(float)s;
    }
    int m_rep=0,m_bd=0,m_ora=0; float mr_rep=0,mr_bd=0;
    for(int r=0;r<N;r++){
        if(std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f)>=1e-3f)m_rep++;
        if(std::fabs(oursBD[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f)>=1e-3f)m_bd++;
        if(std::fabs(ours[r]-rora[r])/(std::fabs(rora[r])+1e-6f)>=1e-3f)m_ora++;
        float a=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(a>mr_rep)mr_rep=a;
        float b=std::fabs(oursBD[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(b>mr_bd)mr_bd=b;
    }
    printf("[iq4_nl] rel>1e-3 repack_leaf vs opp: mism=%d/%d max_rel=%.2e | blockdot vs opp: mism=%d/%d max_rel=%.2e | vs oracle: %d/%d\n",
           m_rep,N,mr_rep,m_bd,N,mr_bd,m_ora,N);
    if(vo){ free(g_flush); return (m_rep==0&&m_bd==0&&m_ora==0)?0:1; }

    std::vector<float> oR(N),oBD(N),oO(N); double tR[64],tBD[64],tO[64];
    run_reps([&](){ weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0((size_t)K,oR.data(),(const uint8_t*)packed.data(),(const uint8_t*)act.data(),(size_t)N); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot((size_t)K,&oBD[r],(const uint8_t*)(W.data()+(size_t)r*nb),(const uint8_t*)act.data(),&nrc1); },tBD,reps);
    run_reps([&](){ for(int r=0;r<N;r++) ggml_vec_dot_iq4_nl_q8_0(K,&oO[r],0,W.data()+(size_t)r*nb,0,act.data(),0,1); },tO,reps);
    double rM=med(tR,reps),bM=med(tBD,reps),oM=med(tO,reps);
    printf("[iq4_nl] COLD K=%d N=%d reps=%d | repack_leaf_ms=%.4f(iqr%.3f) blockdot_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f ratio_blockdot=%.4f\n",
           K,N,reps,rM,riqr(tR,reps),bM,riqr(tBD,reps),oM,riqr(tO,reps),oM/rM,oM/bM);
    free(g_flush); return 0;
}
