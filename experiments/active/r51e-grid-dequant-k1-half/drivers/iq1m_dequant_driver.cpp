// iq1m_dequant_driver.cpp — R5.1-D iq1_m dequantize_row 対拍/計時 driver (standalone leaf).
//
// op = dequantize_row (block_iq1_m -> f32 row · void-return · NO reduction). DUT = our
// board-compiled emitted/owned leaf; OPP/REF = ggml's deployed dequantize_row_iq1_m
// (scalar-source, host-autovec = codegen-lottery opponent, 标量类档). Byte-exact ORACLE =
// an INDEPENDENT ZERO-MODEL recompute from the raw quantized bytes: dl*(grid+delta) is one
// add + one mul (NO fma / NO reassociation), so a correct real-vector emit is byte-exact to
// the scalar reference by construction. [K-5b] three requirements + 3-arm anti-hollow.
//
// argv: <nb(blocks·k=nb*256)> <reps> <seed> <inject> [flush_mb]
//   reps==0 => verify-only (byte-exact + corpus, NO timing)
//   reps>0  => cold N=reps timed (ours-leaf vs ggml dequantize_row_iq1_m), flush each region
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
#define IQ1S_DELTA 0.125f
typedef _Float16 ggml_half;

// block_iq1_m = { uint8 qs[32]@0 ; uint8 qh[16]@32 ; uint16 scales[4]@48 } = 56 bytes.
// NO fp16 d field -- d is the packed iq1m_scale fp16 reconstructed from the 4 scale words.
static const int BLK = 56;

#include "tables/iq1m_grid_table.h"   // weft_iq1m_grid[2048] (uint64; ternary int8 values)

extern "C" void weft_emitc_dequant_iq1_m_kernel_dequant_iq1_m(size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_iq1_m(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_iq1_m_kernel_dequant_iq1_m
#define REF  dequantize_row_iq1_m

static float h2f(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1Fu, man=h&0x3FFu, f;
    if(exp==0){ if(man==0) f=sign; else { exp=127u-15u+1u; while(!(man&0x400u)){man<<=1;exp--;} man&=0x3FFu; f=sign|(exp<<23)|(man<<13);} }
    else if(exp==0x1Fu) f=sign|0x7F800000u|(man<<13);
    else f=sign|((exp-15u+127u)<<23)|(man<<13);
    float r; memcpy(&r,&f,4); return r;
}

// INDEPENDENT ZERO-MODEL oracle — re-decode iq1_m from raw bytes in ggml's exact order:
// reconstruct packed scale fp16, per-group dl1/dl2, 3-high-bit idx, +-delta, then
// dl*(grid[j]+delta) as ONE add + ONE mul. Uses the canonical ternary grid (shared
// constant; ggml crosscheck closes the loop). INJECT arms are OUTPUT injections.
static std::set<int> cov_idx; static int cov_dpos=0, cov_dneg=0;
static void oracle_iq1_m(const uint8_t* xall, float* y, int64_t k){
    int64_t nb = k / QK_K;
    const int8_t* gridb = (const int8_t*) weft_iq1m_grid;
    for(int64_t i=0;i<nb;++i){
        const uint8_t* xb = xall + (size_t)i*BLK;
        uint16_t sc0,sc1,sc2,sc3;
        memcpy(&sc0,xb+48,2); memcpy(&sc1,xb+50,2); memcpy(&sc2,xb+52,2); memcpy(&sc3,xb+54,2);
        uint16_t scbits = (uint16_t)((sc0>>12) | ((sc1>>8)&0x00f0) | ((sc2>>4)&0x0f00) | (sc3&0xf000));
        float d = h2f(scbits);
        uint16_t sc[4] = {sc0,sc1,sc2,sc3};
        float* yy = y + (size_t)i*QK_K;
        for(int ib=0; ib<8; ++ib){
            float dl1 = d*(float)(2*((sc[ib/2]>>(6*(ib%2)+0))&0x7)+1);
            float dl2 = d*(float)(2*((sc[ib/2]>>(6*(ib%2)+3))&0x7)+1);
            int qh0 = xb[32+2*ib+0], qh1 = xb[32+2*ib+1];
            int qs0=xb[4*ib+0], qs1=xb[4*ib+1], qs2=xb[4*ib+2], qs3=xb[4*ib+3];
            int idx[4] = { qs0 | ((qh0<<8)&0x700), qs1 | ((qh0<<4)&0x700),
                           qs2 | ((qh1<<8)&0x700), qs3 | ((qh1<<4)&0x700) };
            float delta[4] = { (qh0&0x08)?-IQ1S_DELTA:IQ1S_DELTA, (qh0&0x80)?-IQ1S_DELTA:IQ1S_DELTA,
                               (qh1&0x08)?-IQ1S_DELTA:IQ1S_DELTA, (qh1&0x80)?-IQ1S_DELTA:IQ1S_DELTA };
            for(int l=0;l<4;++l){
                float dl = (l<2)?dl1:dl2;
                const int8_t* grid = gridb + (size_t)idx[l]*8;
                cov_idx.insert(idx[l]); if(delta[l]<0) cov_dneg=1; else cov_dpos=1;
                float* yg = yy + ib*32 + l*8;
                for(int j=0;j<8;++j)
                    yg[j] = dl * ((float)grid[j] + delta[l]);
            }
        }
    }
}

// BIT-exact compare (byte-exact ZERO-MODEL): reinterpret both floats and compare the 32
// bits. Correct under NaN (identical qNaN bits == equal; iq1_m's reconstructed packed
// scale can be Inf/NaN for some random nibble patterns, and RISC-V produces the canonical
// qNaN identically in ours/ggml/oracle) and -0/+0 (same sign of zero in all three).
static inline bool bne(float a, float b){ uint32_t x,y; memcpy(&x,&a,4); memcpy(&y,&b,4); return x!=y; }
static inline void bitflip(float* p){ uint32_t b; memcpy(&b,p,4); b^=1u; memcpy(p,&b,4); }

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
    printf("=== iq1_m dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n", nb,(long long)k,seed,INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb*BLK);
    for(int i=0;i<nb;++i){
        uint8_t* xb=&blocks[(size_t)i*BLK];
        for(int b=0;b<BLK;++b) xb[b]=(uint8_t)(rng()&0xff);
        // keep the reconstructed scale exponent in a sane fp16 range (avoid inf/nan blowups
        // in the cold-timing buffers); the scale nibbles are the high 4 bits of each word.
        // (correctness holds for any bytes; this only tames the magnitude for timing.)
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
    REF(blocks.data(), ref.data(), k);
    oracle_iq1_m(blocks.data(), ora.data(), k);
    if(INJECT==1) bitflip(&ours[k/2]);
    if(INJECT==2) bitflip(&ora [k/2]);
    if(INJECT==3) bitflip(&ref [k/2]);

    long mOO=0,mRO=0,mOG=0; int firstOO=-1;
    for(int64_t i=0;i<k;++i){
        if(bne(ours[i],ora[i])){ if(firstOO<0)firstOO=(int)i; mOO++; }
        if(bne(ref [i],ora[i])) mRO++;
        if(bne(ours[i],ref[i])) mOG++;
    }
    bool covok=(cov_idx.size()>=2000 && cov_dpos && cov_dneg); // grid idx near-full 2048; both delta signs
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n", mOO,(long long)k,firstOO, mOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n", mRO,(long long)k, mRO==0?"PASS":"VOID-ORACLE");
    printf("  CROSSCHECK ours-vs-ggml  : mism=%ld/%lld -> %s\n", mOG,(long long)k, mOG==0?"PASS":"MISMATCH");
    printf("  CORPUS grid_idx=%zu/2048 delta{+%d,-%d} -> %s\n", cov_idx.size(), cov_dpos, cov_dneg, covok?"COMPLETE":"INCOMPLETE-CORPUS");

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
    printf("DEQUANT_ROW fmt=iq1_m nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,(unsigned long long)g_sink);
    free(g_flush); return 0;
}
