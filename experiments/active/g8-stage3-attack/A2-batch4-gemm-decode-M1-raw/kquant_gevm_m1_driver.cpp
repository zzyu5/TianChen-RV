// A2-batch4 K-quant M=1 GEVM (decode) paired A/B driver (q4_K first).
//  OURS = repack-GEVM LEAF: ONE call, weight=block_q4_Kx16 (stride 2304), act=PLAIN block_q8_K (292B, M=1).
//  OPP  = stock ggml_vec_dot_q4_K_q8_K per-column (block-dot @ M=1).
//  GATE = OURS-vs-OPP agreement (rel<1e-3).  NOTE: this is TWO independent codegens agreeing
//         (our repack leaf vs ggml block-dot), NOT a from-scratch ZERO-MODEL scalar oracle
//         (no independent q4_K super-block scalar decode authored this round -- labeled honestly).
//  Weight repack (pack_w, get_scale_min_k4) reused byte-exact from kquant_repack_verify_q4K.c.
//  argv: <fmt q4_K> <K(mult256)> <N=nc(mult16)> <reps> [verify_only=0] [seed]
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

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
struct block_q4_K { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; }; // 144
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };                    // 292
static_assert(sizeof(block_q4_K)==144,"q4_K"); static_assert(sizeof(block_q8_K)==292,"q8_K");

extern "C" void weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
extern "C" void ggml_vec_dot_q4_K_q8_K(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);

static inline void wr16(uint8_t* p, ggml_half v){ memcpy(p,&v,2); }
static void get_scale_min_k4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; } else { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); } }
// weight [nc][nb] -> block_q4_Kx16 [grp_c][nb] stride 2304 (byte-exact to kquant_repack_verify_q4K.c)
static void pack_w(std::vector<uint8_t>&W,const std::vector<block_q4_K>&o,int nc,int nb){
    int ng=nc/16; W.assign((size_t)ng*nb*2304,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*2304];
        for(int c=0;c<16;++c){ const block_q4_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,x.d); wr16(blk+32+2*c,x.dmin);
            uint8_t sc[8],mn[8]; for(int jj=0;jj<8;++jj) get_scale_min_k4(jj,x.scales,&sc[jj],&mn[jj]);
            for(int gg=0;gg<8;++gg){ int lo=(gg<4)?(64+gg*16+c):(128+(gg-4)*16+c); blk[lo]=(uint8_t)(((mn[gg]&0xF)<<4)|(sc[gg]&0xF)); }
            for(int sb=0;sb<4;++sb) blk[192+sb*16+c]=(uint8_t)(((sc[sb]>>4)&3)|(((mn[sb]>>4)&3)<<2)|(((sc[sb+4]>>4)&3)<<4)|(((mn[sb+4]>>4)&3)<<6));
            for(int i=0;i<128;++i) blk[256+i*16+c]=x.qs[i];
        } }
}

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static const size_t FLUSH_BYTES=32ull*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&&f,double*out,int reps){ for(int r=0;r<reps;r++){ cold_flush(); double a=now_ms(); f(); out[r]=now_ms()-a; } }
static double med(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

static int test_q4_K(int N,int K,bool do_perf,int reps,unsigned seed){
    int nb=K/QK_K; std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(0,255); std::uniform_real_distribution<float> dd(0.001f,0.05f), dmn(0.0f,0.01f);
    auto f16=[&](float x){ ggml_half h; float xx=x; /* store via _Float16-free path: use ggml round approx */
        // simple f32->f16 round-to-nearest-even
        uint32_t X; memcpy(&X,&xx,4); uint32_t sign=(X>>16)&0x8000u; int32_t e=(int32_t)((X>>23)&0xff)-127+15; uint32_t man=X&0x7fffffu; uint16_t r;
        if(e<=0){ r=(uint16_t)sign; } else if(e>=31){ r=(uint16_t)(sign|0x7c00u); } else { r=(uint16_t)(sign|((uint32_t)e<<10)|(man>>13)); if((man>>12)&1)r++; } memcpy(&h,&r,2); return h; };
    std::vector<block_q4_K> W((size_t)N*nb);
    for(auto&x:W){ x.d=f16(dd(rng)); x.dmin=f16(dmn(rng)); for(int s=0;s<12;s++)x.scales[s]=(uint8_t)qd(rng); for(int i=0;i<128;i++)x.qs[i]=(uint8_t)qd(rng); }
    block_q8_K act; { act.d=dd(rng); std::uniform_int_distribution<int> ad(-90,90);
        for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++){ int v=ad(rng); act.qs[g*16+i]=(int8_t)v; s+=v; } act.bsums[g]=(int16_t)s; } }
    std::vector<uint8_t> Wr; pack_w(Wr,W,N,nb);

    std::vector<float> ours(N,0.f), ropp(N,0.f);
    weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)K,ours.data(),Wr.data(),(const uint8_t*)&act,(size_t)N);
    for(int r=0;r<N;r++) ggml_vec_dot_q4_K_q8_K(K,&ropp[r],0,W.data()+(size_t)r*nb,0,&act,0,1);
    int m_opp=0; float mr=0; for(int r=0;r<N;r++){ float a=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(a>=1e-3f)m_opp++; if(a>mr)mr=a; }
    printf("[q4_K] GATE K=%d N=%d seed=0x%x | repack_leaf vs opp(ggml): mism(rel>=1e-3)=%d/%d maxrel=%.2e\n",K,N,seed,m_opp,N,mr);
    int ok=(m_opp==0);
    if(!do_perf) return ok?0:1;
    std::vector<float> oR(N),oO(N); double tR[80],tO[80];
    run_reps([&](){ weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)K,oR.data(),Wr.data(),(const uint8_t*)&act,(size_t)N); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) ggml_vec_dot_q4_K_q8_K(K,&oO[r],0,W.data()+(size_t)r*nb,0,&act,0,1); },tO,reps);
    double rM=med(tR,reps),oM=med(tO,reps);
    printf("[q4_K] COLD K=%d N=%d reps=%d seed=0x%x | repack_leaf_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f %s\n",
           K,N,reps,seed,rM,reliqr(tR,reps),oM,reliqr(tO,reps),oM/rM,(oM/rM>=0.8?"PASS":"named-X"));
    return ok?0:1;
}

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s <q4_K> K N reps [verify_only] [seed]\n",argv[0]); return 2; }
    const char* fmt=argv[1]; int K=atoi(argv[2]),N=atoi(argv[3]),reps=atoi(argv[4]);
    bool vo=argc>5&&atoi(argv[5])!=0; if(reps>80)reps=80; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    if(K%256)K=(K/256)*256; if(K<256)K=256;
    unsigned seed=argc>6?(unsigned)strtoul(argv[6],0,0):0xC0FFEE1u;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); if(!g_flush){fprintf(stderr,"OOM\n");return 3;} memset(g_flush,1,FLUSH_BYTES);
    int rc=1;
    if(!strcmp(fmt,"q4_K")) rc=test_q4_K(N,K,!vo,reps,seed);
    else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }
    free(g_flush); return rc;
}
