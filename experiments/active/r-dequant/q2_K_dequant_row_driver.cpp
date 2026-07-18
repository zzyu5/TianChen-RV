// q2_K_dequant_row_driver.cpp — R线 dequantize_row q2_K 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_q2_K -> f32 row · void-return · NO reduction). DUT = our OWNED
// real-vector leaf (per 16-lane group: vle8 + vsrl(shift)+vand(3) the 2-bit quant + vzext_vf4
// + vfcvt + vfmv_v_f(ml)+vfmsac_vf(dl) FUSED dl*q - ml + vse32); OPP/REF = ggml deployed
// dequantize_row_q2_K (标量类档). Byte-exact ORACLE = INDEPENDENT ZERO-MODEL recompute
// ([K-5]): q2_K decodes as d=fp16@80, dmin=fp16@82, 4-bit packed (sc&0xF, sc>>4) from
// scales[16]@0, dl=d*(sc&0xF), ml=dmin*(sc>>4), q2=(qs>>shift)&3; y = dl*q2 - ml. The fold is
// a SINGLE fused mul-sub (std::fma) matching the contracted opponent autovec and our vfmsac.
//
// ANTI-HOLLOW: INJECT=1 oracle-fault / INJECT=2 dut-fault / INJECT=3 leaf d-offset flip
// (v8 + 80 -> v8 + 81, harness `cmp -l | wc -l` == 1) => ours-vs-oracle RED ONLY.
// argv: <nb(blocks·k=nb*256)> <reps> <seed> [flush_mb]   (reps==0 => verify-only)
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <time.h>

#ifndef INJECT
#define INJECT 0
#endif
#ifndef FLUSH_MB
#define FLUSH_MB 224
#endif

#define QK_K 256
typedef _Float16 ggml_half;

// block_q2_K = { uint8_t scales[16]; uint8_t qs[64]; fp16 d; fp16 dmin; } = 84 bytes.
static const int BLK = 84;

extern "C" void weft_emitc_dequant_q2_K_kernel_dequant_q2_K(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_q2_K(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_q2_K_kernel_dequant_q2_K
#define REF  dequantize_row_q2_K

static std::set<int> cov;   // (j<<2)|qv  -> want 16 (4 shifts x 4 values)
static void oracle_q2_K(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        const uint8_t* scales = xb + 0;
        const uint8_t* qs = xb + 16;
        float d  = (float)*(const _Float16*)(xb + 80);
        float mn = (float)*(const _Float16*)(xb + 82);
        float* yy = y + (size_t)i * QK_K;
        for (int nn = 0; nn < 2; ++nn) {
            const uint8_t* q = qs + nn * 32;
            for (int j = 0; j < 4; ++j) {
                int shift = 2 * j;
                for (int half = 0; half < 2; ++half) {
                    uint8_t sc = scales[nn * 8 + 2 * j + half];
                    float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        int qv = (q[half * 16 + l] >> shift) & 3;
                        cov.insert((j << 2) | qv);
                        float v = std::fma(dl, (float)qv, -ml);
#if INJECT==1
                        if (i == 0 && nn == 0 && j == 0 && half == 0 && l == 0) v += 1.0f;
#endif
                        *yy++ = v;
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
static double riqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size(); double m=v[n/2];
    return m>0 ? 100.0*(v[3*n/4]-v[n/4])/m : 0.0; }

int main(int argc, char** argv){
    if (argc < 4){ fprintf(stderr,"usage: %s nb reps seed [flush_mb]\n", argv[0]); return 2; }
    int nb   = atoi(argv[1]);
    int reps = atoi(argv[2]);
    unsigned seed = (unsigned)strtoul(argv[3],0,0);
    int64_t k = (int64_t)nb * QK_K;
    printf("=== q2_K dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        for (int b = 0; b < 80; ++b) xb[b] = (uint8_t)(rng() & 0xff); // scales + qs random
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;
        float mf = ((float)(rng() % 2001) - 1000.f) / 4000.f;
        _Float16 dh = (_Float16)df; memcpy(xb + 80, &dh, 2);
        _Float16 mh = (_Float16)mf; memcpy(xb + 82, &mh, 2);
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_q2_K(blocks.data(), ora.data(), k);

    long mismOO = 0, mismRO = 0; int firstOO = -1;
    for (int64_t i = 0; i < k; ++i) {
        if (!(ours[i] == ora[i])) { if (firstOO<0) firstOO=(int)i; mismOO++; }
        if (!(ref[i]  == ora[i])) mismRO++;
    }
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n",
           mismOO, (long long)k, firstOO, mismOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n",
           mismRO, (long long)k, mismRO==0?"PASS":"VOID-ORACLE");

    bool covok = (cov.size()==16);
    printf("  CORPUS shift_val=%zu/16 -> %s\n", cov.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

    if (reps == 0) {
        bool ok = (mismOO==0 && mismRO==0);
        if (INJECT==0 && !covok) { fprintf(stderr,"INCOMPLETE-CORPUS: raise nb\n"); return 9; }
        return ok ? 0 : 1;
    }
    if (mismOO != 0) {
        fprintf(stderr,"VOID-CORRECTNESS: ours mismatched oracle => timing forbidden\n");
        return 1;
    }

    g_flush=(uint8_t*)aligned_alloc(64,FLUSHB); if(!g_flush){ fprintf(stderr,"OOM flush\n"); return 3; }
    memset(g_flush,1,FLUSHB);
    std::vector<double> tR, tG;
    std::vector<float> bR(k), bG(k);
    for (int p=0;p<reps;++p){
        cold_flush(); double t0=now_ns();
        LEAF((size_t)k, blocks.data(), bR.data());
        tR.push_back(now_ns()-t0);
        cold_flush(); double t1=now_ns();
        REF(blocks.data(), bG.data(), k);
        tG.push_back(now_ns()-t1);
    }
    double rM=med(tR), gM=med(tG);
    printf("REPS_OURS seed=0x%X :",seed); for(double v:tR) printf(" %.0f",v); printf("\n");
    printf("REPS_GGML seed=0x%X :",seed); for(double v:tG) printf(" %.0f",v); printf("\n");
    printf("DEQUANT_ROW fmt=q2_K nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
