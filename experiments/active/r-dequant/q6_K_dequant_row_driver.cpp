// q6_K_dequant_row_driver.cpp — R线 dequantize_row q6_K 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_q6_K -> f32 row · void-return · NO reduction). DUT = our OWNED
// real-vector leaf (per 16-lane group: vle8 ql + vle8 qh + the nibble (lo/hi) + the qh 2-bit
// high term -> combined 0..63 -> vsub 32 -> vfcvt -> vfmul_vf(dsc) SINGLE mul + vse32);
// OPP/REF = ggml deployed dequantize_row_q6_K (标量类档). Byte-exact ORACLE = INDEPENDENT
// ZERO-MODEL recompute ([K-5]): q6_K decodes as d=fp16@208, SIGNED int8 scales[16]@192, q =
// ((ql nibble) | ((qh 2 bits)<<4)) - 32; y = (d*sc[is]) * q. Two f32 muls, NO add (NO
// fp-contraction ambiguity) so no fma needed; matches ggml's `d * sc * q` left-assoc.
//
// ANTI-HOLLOW: INJECT=1 oracle-fault / INJECT=2 dut-fault / INJECT=3 leaf d-offset flip
// (v8 + 208 -> v8 + 207, harness `cmp -l | wc -l` == 1) => ours-vs-oracle RED ONLY.
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

// block_q6_K = { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; fp16 d; } = 210 bytes.
static const int BLK = 210;

extern "C" void weft_emitc_dequant_q6_K_kernel_dequant_q6_K(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_q6_K(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_q6_K_kernel_dequant_q6_K
#define REF  dequantize_row_q6_K

static std::set<int> cov;   // combined 6-bit value 0..63 -> want 64
static void oracle_q6_K(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d = (float)*(const _Float16*)(xb + 208);
        const uint8_t* ql = xb + 0;
        const uint8_t* qh = xb + 128;
        const int8_t*  sc = (const int8_t*)(xb + 192);
        float* yy = y + (size_t)i * QK_K;
        for (int nn = 0; nn < 2; ++nn) {
            const uint8_t* qlp = ql + nn * 64;
            const uint8_t* qhp = qh + nn * 32;
            const int8_t*  scp = sc + nn * 8;
            float* yb = yy + nn * 128;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int c1 = (qlp[l]      & 0xF) | (((qhp[l] >> 0) & 3) << 4);
                int c2 = (qlp[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4);
                int c3 = (qlp[l]      >> 4)  | (((qhp[l] >> 4) & 3) << 4);
                int c4 = (qlp[l + 32] >> 4)  | (((qhp[l] >> 6) & 3) << 4);
                cov.insert(c1); cov.insert(c2); cov.insert(c3); cov.insert(c4);
                int q1 = c1 - 32, q2 = c2 - 32, q3 = c3 - 32, q4 = c4 - 32;
                float v1 = (d * scp[is + 0]) * (float)q1;
#if INJECT==1
                if (i == 0 && nn == 0 && l == 0) v1 += 1.0f;
#endif
                yb[l +  0] = v1;
                yb[l + 32] = (d * scp[is + 2]) * (float)q2;
                yb[l + 64] = (d * scp[is + 4]) * (float)q3;
                yb[l + 96] = (d * scp[is + 6]) * (float)q4;
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
    printf("=== q6_K dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        for (int b = 0; b < 208; ++b) xb[b] = (uint8_t)(rng() & 0xff); // ql + qh + scales
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;
        _Float16 dh = (_Float16)df; memcpy(xb + 208, &dh, 2);
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_q6_K(blocks.data(), ora.data(), k);

    long mismOO = 0, mismRO = 0; int firstOO = -1;
    for (int64_t i = 0; i < k; ++i) {
        if (!(ours[i] == ora[i])) { if (firstOO<0) firstOO=(int)i; mismOO++; }
        if (!(ref[i]  == ora[i])) mismRO++;
    }
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n",
           mismOO, (long long)k, firstOO, mismOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n",
           mismRO, (long long)k, mismRO==0?"PASS":"VOID-ORACLE");

    bool covok = (cov.size()==64);
    printf("  CORPUS combined6=%zu/64 -> %s\n", cov.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

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
    printf("DEQUANT_ROW fmt=q6_K nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
