// iq3s_dequant_driver.cpp — R5.1-C iq3_s dequantize_row対拍/計時 driver (standalone leaf).
//
// op = dequantize_row (block_iq3_s -> f32 row · void-return · NO reduction). DUT = our
// board-compiled emitted/narrow leaf; OPP/REF = ggml's deployed dequantize_row_iq3_s
// (scalar-source, host-autovec = codegen-lottery opponent, 标量类档). Byte-exact ORACLE =
// an INDEPENDENT ZERO-MODEL recompute from the raw quantized bytes (dequant has NO fp
// reassociation — the ONLY rounding is db*grid, sign fold multiplies by EXACT +-1.0f — so
// a correct real-vector emit is byte-exact to the scalar reference by construction, and the
// board `==` compare is the certificate). [K-5b] three requirements + 3-arm anti-hollow.
//
// argv: <nb(blocks·k=nb*256)> <reps> <seed> <inject> [flush_mb]
//   reps==0 => verify-only (byte-exact + corpus, NO timing)
//   reps>0  => cold N=reps timed (ours-leaf vs ggml dequantize_row_iq3_s), flush each region
//   inject: 0 clean · 1 ours-out fault · 2 oracle-out fault · 3 ggml-out fault (3-arm)
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <time.h>

#ifndef FLUSH_MB
#define FLUSH_MB 224
#endif
#define QK_K 256
typedef _Float16 ggml_half;

// block_iq3_s = { fp16 d@0 ; uint8 qs[64]@2 ; uint8 qh[8]@66 ; uint8 signs[32]@74 ;
//                 uint8 scales[4]@106 } = 110 bytes.
static const int BLK = 110;

#include "tables/iq3s_grid_table.h"   // weft_iq3s_grid[512] (uint32) — the ONE permitted share

extern "C" void weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s(size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_iq3_s(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s
#define REF  dequantize_row_iq3_s

// ---- pure-integer fp16 -> fp32 (independent of ggml) ----
static float h2f(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1Fu, man=h&0x3FFu, f;
    if(exp==0){ if(man==0) f=sign; else { exp=127u-15u+1u; while(!(man&0x400u)){man<<=1;exp--;} man&=0x3FFu; f=sign|(exp<<23)|(man<<13);} }
    else if(exp==0x1Fu) f=sign|0x7F800000u|(man<<13);
    else f=sign|((exp-15u+127u)<<23)|(man<<13);
    float r; memcpy(&r,&f,4); return r;
}

// INDEPENDENT ZERO-MODEL oracle — re-decode iq3_s from raw bytes in ggml's exact order
// (single rounding db*grid; +-1.0f sign exact). INJECT arms are OUTPUT injections (below).
static std::set<int> cov_idx, cov_sig;
static void oracle_iq3_s(const uint8_t* xall, float* y, int64_t k){
    int64_t nb = k / QK_K;
    const uint8_t* gridb = (const uint8_t*) weft_iq3s_grid;
    for(int64_t i=0;i<nb;++i){
        const uint8_t* xb = xall + (size_t)i*BLK;
        uint16_t dh; memcpy(&dh,xb,2); float d = h2f(dh);
        float* yy = y + (size_t)i*QK_K;
        for(int g=0; g<4; ++g){
            int scByte = xb[106+g];
            float db[2] = { d*(float)(1+2*(scByte&0xf)), d*(float)(1+2*(scByte>>4)) };
            for(int p=0;p<2;++p){
                int qh = xb[66 + 2*g + p];
                float dbp = db[p];
                const uint8_t* qs = xb + 2 + 16*g + 8*p;
                const uint8_t* sg = xb + 74 + 8*g + 4*p;
                float* yg = yy + g*64 + 32*p;
                for(int l=0;l<4;++l){
                    int idx1 = qs[2*l+0] | ((qh << (8-2*l)) & 256);
                    int idx2 = qs[2*l+1] | ((qh << (7-2*l)) & 256);
                    uint8_t sb = sg[l];
                    cov_idx.insert(idx1); cov_idx.insert(idx2); cov_sig.insert(sb);
                    for(int j=0;j<4;++j){
                        float s0 = (sb>>j)&1 ? -1.f : 1.f;
                        yg[l*8 + j]   = dbp * (float)gridb[idx1*4 + j] * s0;
                        float s1 = (sb>>(j+4))&1 ? -1.f : 1.f;
                        yg[l*8 + 4+j] = dbp * (float)gridb[idx2*4 + j] * s1;
                    }
                }
            }
        }
    }
}

static const size_t FLUSHB = (size_t)FLUSH_MB*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSHB;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double riqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size(); double m=v[n/2]; return m>0?100.0*(v[3*n/4]-v[n/4])/m:0.0; }

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"usage: %s nb reps seed inject [flush_mb]\n",argv[0]); return 2; }
    int nb=atoi(argv[1]), reps=atoi(argv[2]); unsigned seed=(unsigned)strtoul(argv[3],0,0);
    int INJECT=atoi(argv[4]);
    int64_t k=(int64_t)nb*QK_K;
    printf("=== iq3_s dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n", nb,(long long)k,seed,INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb*BLK);
    for(int i=0;i<nb;++i){
        uint8_t* xb=&blocks[(size_t)i*BLK];
        float df=((float)(rng()%2001)-1000.f)/2000.f; _Float16 dh=(_Float16)df; memcpy(xb,&dh,2);
        for(int b=2;b<BLK;++b) xb[b]=(uint8_t)(rng()&0xff);
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
    REF(blocks.data(), ref.data(), k);
    oracle_iq3_s(blocks.data(), ora.data(), k);
    if(INJECT==1) ours[k/2]+=1.0f;   // DUT out fault  -> ours-vs-{oracle,ggml} MUST flip
    if(INJECT==2) ora [k/2]+=1.0f;   // oracle out fault-> {ours,ggml}-vs-oracle MUST flip
    if(INJECT==3) ref [k/2]+=1.0f;   // ggml out fault  -> {ours,oracle}-vs-ggml MUST flip

    long mOO=0,mRO=0,mOG=0; int firstOO=-1;
    for(int64_t i=0;i<k;++i){
        if(!(ours[i]==ora[i])){ if(firstOO<0)firstOO=(int)i; mOO++; }
        if(!(ref [i]==ora[i])) mRO++;
        if(!(ours[i]==ref[i])) mOG++;
    }
    bool covok=(cov_idx.size()>=440 && cov_sig.size()==256); // grid 512 sparse (9th-bit); signs full
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n", mOO,(long long)k,firstOO, mOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n", mRO,(long long)k, mRO==0?"PASS":"VOID-ORACLE");
    printf("  CROSSCHECK ours-vs-ggml  : mism=%ld/%lld -> %s\n", mOG,(long long)k, mOG==0?"PASS":"MISMATCH");
    printf("  CORPUS grid_idx=%zu/512 sign_byte=%zu/256 -> %s\n", cov_idx.size(), cov_sig.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

    if(INJECT!=0){
        bool bites = (mOO!=0) || (mRO!=0) || (mOG!=0);
        printf("  ANTIHOLLOW inject=%d -> %s\n", INJECT, bites?"BITES-OK":"HOLLOW-FAIL");
        return bites?0:3;
    }
    if(reps==0){ bool ok=(mOO==0&&mRO==0&&mOG==0); if(!covok){fprintf(stderr,"INCOMPLETE-CORPUS: raise nb\n"); return 9;} return ok?0:1; }
    if(mOO!=0){ fprintf(stderr,"VOID-CORRECTNESS: timing forbidden\n"); return 1; }

    g_flush=(uint8_t*)aligned_alloc(64,FLUSHB); if(!g_flush){return 3;} memset(g_flush,1,FLUSHB);
    std::vector<double> tR,tG; std::vector<float> bR(k),bG(k);
    for(int p=0;p<reps;++p){
        cold_flush(); double t0=now_ns(); LEAF((size_t)k,blocks.data(),bR.data()); tR.push_back(now_ns()-t0);
        cold_flush(); double t1=now_ns(); REF(blocks.data(),bG.data(),k);          tG.push_back(now_ns()-t1);
    }
    double rM=med(tR),gM=med(tG);
    printf("DEQUANT_ROW fmt=iq3_s nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,(unsigned long long)g_sink);
    free(g_flush); return 0;
}
