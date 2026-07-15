// A2-batch5 K-quant M=1 GEVM (decode) paired A/B driver: q2_K / q3_K / q5_K / q6_K.
//  OURS = repack-GEVM LEAF (body = typed_repack_gemv_loop_body, front-door
//         --weft-rvv-lower-quant-contraction/--weft-rvv-lower-to-emitc product; NOT vec_dot block-dot).
//         ONE call over nc columns, weight = block_qX_Kx16 (per-format stride), act = PLAIN block_q8_K (292B, M=1).
//  OPP  = stock ggml_vec_dot_qX_K_q8_K per-column (block-dot @ M=1) from board libggml-cpu.so.
//  GATE = OURS-vs-OPP agreement (rel<1e-3): TWO independent codegens (our repack leaf vs stock ggml block-dot)
//         reading DISJOINT layouts (packed x16 vs original per-block) agreeing => (pack_w, leaf) both correct.
//         NOT a from-scratch ZERO-MODEL scalar oracle (labeled honestly, same gate class as batch4 q4_K).
//  Weight repack (pack_w) byte-exact to tools/e2e-harness/board/kquant_repack_verify_q{2,3,6}K.c
//  + q5_K derived from rvv-to-emitc-repack-gemv-q5-K-q8-K.mlir layout (q4_K scale packing + qh@256 + qs@768).
//  argv: <fmt q2_K|q3_K|q5_K|q6_K> <K(mult256)> <N=nc(mult16)> <reps> [verify_only=0] [seed]
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

// ---- canonical ggml block structs (opponent ggml_vec_dot reads these) ----
struct block_q2_K { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; ggml_half d, dmin; };          // 84
struct block_q3_K { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d; };// 110
struct block_q5_K { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; }; // 176
struct block_q6_K { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; };// 210
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };                          // 292
static_assert(sizeof(block_q2_K)==84,"q2_K"); static_assert(sizeof(block_q3_K)==110,"q3_K");
static_assert(sizeof(block_q5_K)==176,"q5_K"); static_assert(sizeof(block_q6_K)==210,"q6_K");
static_assert(sizeof(block_q8_K)==292,"q8_K");

// ---- leaf ABI (all 5-arg (n,s,vx,vy,nc), identical to q4_K batch4) ----
extern "C" void weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t);
extern "C" void weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t);
extern "C" void weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t);
extern "C" void weft_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t);
// ---- opponent stock ggml block-dot ----
extern "C" void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern "C" void ggml_vec_dot_q3_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern "C" void ggml_vec_dot_q5_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern "C" void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static inline void wr16(uint8_t* p, ggml_half v){ memcpy(p,&v,2); }
// f32 -> f16 round-to-nearest-even (libcall-free)
static ggml_half f16(float xx){ uint32_t X; memcpy(&X,&xx,4); uint32_t sign=(X>>16)&0x8000u;
    int32_t e=(int32_t)((X>>23)&0xff)-127+15; uint32_t man=X&0x7fffffu; uint16_t r;
    if(e<=0){ r=(uint16_t)sign; } else if(e>=31){ r=(uint16_t)(sign|0x7c00u); }
    else { r=(uint16_t)(sign|((uint32_t)e<<10)|(man>>13)); if((man>>12)&1)r++; } return r; }
static void get_scale_min_k4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; } else { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); } }
// unpack 12 packed q3_K scale bytes -> 16 SIGNED int8 (value-32), ggml bit-dance (from kquant_repack_verify_q3K.c)
static void unpack_q3K_scales(const uint8_t packed[12],int8_t out[16]){
    const uint32_t k1=0x03030303u,k2=0x0f0f0f0fu; uint32_t aux[4]; memcpy(aux,packed,12);
    uint32_t t=aux[2];
    aux[2]=((aux[0]>>4)&k2)|(((t>>4)&k1)<<4); aux[3]=((aux[1]>>4)&k2)|(((t>>6)&k1)<<4);
    aux[0]=((aux[0])&k2)|(((t>>0)&k1)<<4);    aux[1]=((aux[1])&k2)|(((t>>2)&k1)<<4);
    const int8_t* sc=(const int8_t*)aux; for(int i=0;i<16;++i)out[i]=(int8_t)((int)sc[i]-32);
}

// ---- per-format pack_w: weight [nc][nb] -> block_qX_Kx16 [grp_c][nb] (byte-exact to repack-verify / MLIR attrs) ----
static void pack_q2K(std::vector<uint8_t>&W,const std::vector<block_q2_K>&o,int nc,int nb){ // stride 1344
    int ng=nc/16; W.assign((size_t)ng*nb*1344,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*1344];
        for(int c=0;c<16;++c){ const block_q2_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,x.d); wr16(blk+32+2*c,x.dmin);
            for(int s=0;s<16;++s) blk[64+s*16+c]=x.scales[s];
            for(int i=0;i<64;++i) blk[320+i*16+c]=x.qs[i]; } }
}
static void pack_q3K(std::vector<uint8_t>&W,const std::vector<block_q3_K>&o,int nc,int nb){ // stride 1824
    int ng=nc/16; W.assign((size_t)ng*nb*1824,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*1824];
        for(int c=0;c<16;++c){ const block_q3_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,x.d);
            int8_t sc[16]; unpack_q3K_scales(x.scales,sc);
            for(int s=0;s<16;++s) blk[32+s*16+c]=(uint8_t)sc[s];
            for(int i=0;i<32;++i) blk[288+i*16+c]=x.hmask[i];
            for(int i=0;i<64;++i) blk[800+i*16+c]=x.qs[i]; } }
}
static void pack_q5K(std::vector<uint8_t>&W,const std::vector<block_q5_K>&o,int nc,int nb){ // stride 2816
    int ng=nc/16; W.assign((size_t)ng*nb*2816,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*2816];
        for(int c=0;c<16;++c){ const block_q5_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,x.d); wr16(blk+32+2*c,x.dmin);
            uint8_t sc[8],mn[8]; for(int jj=0;jj<8;++jj) get_scale_min_k4(jj,x.scales,&sc[jj],&mn[jj]);
            for(int gg=0;gg<8;++gg){ int lo=(gg<4)?(64+gg*16+c):(128+(gg-4)*16+c); blk[lo]=(uint8_t)(((mn[gg]&0xF)<<4)|(sc[gg]&0xF)); }
            for(int sb=0;sb<4;++sb) blk[192+sb*16+c]=(uint8_t)(((sc[sb]>>4)&3)|(((mn[sb]>>4)&3)<<2)|(((sc[sb+4]>>4)&3)<<4)|(((mn[sb+4]>>4)&3)<<6));
            for(int i=0;i<32;++i) blk[256+i*16+c]=x.qh[i];   // qh 5th-bit plane @256
            for(int i=0;i<128;++i) blk[768+i*16+c]=x.qs[i];  // nibbles @768
        } }
}
static void pack_q6K(std::vector<uint8_t>&W,const std::vector<block_q6_K>&o,int nc,int nb){ // stride 3360
    int ng=nc/16; W.assign((size_t)ng*nb*3360,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*3360];
        for(int c=0;c<16;++c){ const block_q6_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,x.d);
            for(int s=0;s<16;++s) blk[32+s*16+c]=(uint8_t)x.scales[s];
            for(int i=0;i<64;++i) blk[288+i*16+c]=x.qh[i];
            for(int i=0;i<128;++i) blk[1312+i*16+c]=x.ql[i]; } }
}

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static const size_t FLUSH_BYTES=32ull*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
template<typename F> static void run_reps(F&&f,double*out,int reps){ for(int r=0;r<reps;r++){ cold_flush(); double a=now_ms(); f(); out[r]=now_ms()-a; } }
static double med(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double*v,int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

// build a plain M=1 activation block_q8_K (correct bsums), matching q4_K batch4
static void build_act(block_q8_K& act, std::mt19937& rng){
    std::uniform_real_distribution<float> dd(0.001f,0.05f);
    act.d=dd(rng); std::uniform_int_distribution<int> ad(-90,90);
    for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++){ int v=ad(rng); act.qs[g*16+i]=(int8_t)v; s+=v; } act.bsums[g]=(int16_t)s; }
}

template<typename Blk, typename PackFn, typename LeafFn, typename OppFn>
static int run_fmt(const char* name, int N,int K,bool do_perf,int reps,unsigned seed,
                   void(*genblk)(Blk&,std::mt19937&), PackFn pack, LeafFn leaf, OppFn opp){
    int nb=K/QK_K; std::mt19937 rng(seed);
    std::vector<Blk> W((size_t)N*nb);
    for(auto&x:W) genblk(x,rng);
    block_q8_K act; build_act(act,rng);
    std::vector<uint8_t> Wr; pack(Wr,W,N,nb);

    std::vector<float> ours(N,0.f), ropp(N,0.f);
    leaf((size_t)K,ours.data(),Wr.data(),(const uint8_t*)&act,(size_t)N);
    for(int r=0;r<N;r++) opp(K,&ropp[r],0,W.data()+(size_t)r*nb,0,&act,0,1);
    int m_opp=0; float mr=0; for(int r=0;r<N;r++){ float a=std::fabs(ours[r]-ropp[r])/(std::fabs(ropp[r])+1e-6f); if(a>=1e-3f)m_opp++; if(a>mr)mr=a; }
    printf("[%s] GATE K=%d N=%d seed=0x%x | repack_leaf vs opp(ggml): mism(rel>=1e-3)=%d/%d maxrel=%.2e\n",name,K,N,seed,m_opp,N,mr);
    int ok=(m_opp==0);
    if(!do_perf) return ok?0:1;
    std::vector<float> oR(N),oO(N); double tR[80],tO[80];
    run_reps([&](){ leaf((size_t)K,oR.data(),Wr.data(),(const uint8_t*)&act,(size_t)N); },tR,reps);
    run_reps([&](){ for(int r=0;r<N;r++) opp(K,&oO[r],0,W.data()+(size_t)r*nb,0,&act,0,1); },tO,reps);
    double rM=med(tR,reps),oM=med(tO,reps);
    printf("[%s] COLD K=%d N=%d reps=%d seed=0x%x | repack_leaf_ms=%.4f(iqr%.3f) opp_ms=%.4f(iqr%.3f) | ratio_repack=%.4f %s\n",
           name,K,N,reps,seed,rM,reliqr(tR,reps),oM,reliqr(tO,reps),oM/rM,(oM/rM>=0.8?"PASS":"named-X"));
    return ok?0:1;
}

// ---- per-format weight generators (small d/dmin, bounded quants; ggml-canonical semantics) ----
static void gen_q2K(block_q2_K& x,std::mt19937& rng){ std::uniform_int_distribution<int> qd(0,255);
    std::uniform_real_distribution<float> dd(0.001f,0.05f),dmn(0.0f,0.01f);
    for(int s=0;s<16;s++)x.scales[s]=(uint8_t)qd(rng); for(int i=0;i<64;i++)x.qs[i]=(uint8_t)qd(rng);
    x.d=f16(dd(rng)); x.dmin=f16(dmn(rng)); }
static void gen_q3K(block_q3_K& x,std::mt19937& rng){ std::uniform_int_distribution<int> qd(0,255);
    std::uniform_real_distribution<float> dd(0.001f,0.05f);
    for(int i=0;i<32;i++)x.hmask[i]=(uint8_t)qd(rng); for(int i=0;i<64;i++)x.qs[i]=(uint8_t)qd(rng);
    for(int s=0;s<12;s++)x.scales[s]=(uint8_t)qd(rng); x.d=f16(dd(rng)); }
static void gen_q5K(block_q5_K& x,std::mt19937& rng){ std::uniform_int_distribution<int> qd(0,255);
    std::uniform_real_distribution<float> dd(0.001f,0.05f),dmn(0.0f,0.01f);
    for(int s=0;s<12;s++)x.scales[s]=(uint8_t)qd(rng); for(int i=0;i<32;i++)x.qh[i]=(uint8_t)qd(rng);
    for(int i=0;i<128;i++)x.qs[i]=(uint8_t)qd(rng); x.d=f16(dd(rng)); x.dmin=f16(dmn(rng)); }
static void gen_q6K(block_q6_K& x,std::mt19937& rng){ std::uniform_int_distribution<int> qd(0,255);
    std::uniform_int_distribution<int> sd(-32,31); std::uniform_real_distribution<float> dd(0.001f,0.05f);
    for(int i=0;i<128;i++)x.ql[i]=(uint8_t)qd(rng); for(int i=0;i<64;i++)x.qh[i]=(uint8_t)qd(rng);
    for(int s=0;s<16;s++)x.scales[s]=(int8_t)sd(rng); x.d=f16(dd(rng)); }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s <q2_K|q3_K|q5_K|q6_K> K N reps [verify_only] [seed]\n",argv[0]); return 2; }
    const char* fmt=argv[1]; int K=atoi(argv[2]),N=atoi(argv[3]),reps=atoi(argv[4]);
    bool vo=argc>5&&atoi(argv[5])!=0; if(reps>80)reps=80; if(reps<1)reps=1; if(N%16)N=(N/16)*16; if(N<16)N=16;
    if(K%256)K=(K/256)*256; if(K<256)K=256;
    unsigned seed=argc>6?(unsigned)strtoul(argv[6],0,0):0xC0FFEE1u;
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); if(!g_flush){fprintf(stderr,"OOM\n");return 3;} memset(g_flush,1,FLUSH_BYTES);
    int rc=1;
    if(!strcmp(fmt,"q2_K")) rc=run_fmt<block_q2_K>("q2_K",N,K,!vo,reps,seed,gen_q2K,pack_q2K,
        weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K,ggml_vec_dot_q2_K_q8_K);
    else if(!strcmp(fmt,"q3_K")) rc=run_fmt<block_q3_K>("q3_K",N,K,!vo,reps,seed,gen_q3K,pack_q3K,
        weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K,ggml_vec_dot_q3_K_q8_K);
    else if(!strcmp(fmt,"q5_K")) rc=run_fmt<block_q5_K>("q5_K",N,K,!vo,reps,seed,gen_q5K,pack_q5K,
        weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K,ggml_vec_dot_q5_K_q8_K);
    else if(!strcmp(fmt,"q6_K")) rc=run_fmt<block_q6_K>("q6_K",N,K,!vo,reps,seed,gen_q6K,pack_q6K,
        weft_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K,ggml_vec_dot_q6_K_q8_K);
    else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }
    free(g_flush); return rc;
}
