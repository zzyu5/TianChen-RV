// q5_1_dequant_row_driver.cpp — R线 dequantize_row q5_1 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_q5_1 -> f32 row · void-return · NO reduction). DUT = our
// board-compiled emitted leaf (OWNED real-vector: vle8 nibble split + vzext + qh 5th-bit
// spread + vfcvt + the FUSED min-add vfmv_v_f(m) + vfmacc_vf(d) + vse32); OPP/REF = ggml's
// deployed dequantize_row_q5_1 (scalar-source, host-autovec = the codegen-lottery opponent,
// 标量类档). Byte-exact ORACLE = an INDEPENDENT ZERO-MODEL recompute from the raw bytes
// ([K-5]): q5_1 folds the 5th bit from the byte-assembled uint32 qh then `y = x*d + m` — a
// SINGLE FUSED multiply-add (one rounding). The -ffp-contract=on opponent autovec's it to
// vfmadd, so the byte-exact reference is the fused form; the oracle uses std::fma to nail
// the single fused rounding UNAMBIGUOUSLY, and the DUT's vfmacc computes the same `m + d*x`.
// The ggml-vs-oracle crosscheck is the FMA-contract consistency probe.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle re-derives d/m/qh/nibbles from the raw bytes.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE.
//   (1) CORPUS COMPLETENESS: random bytes sweep every low/high nibble (0..15) AND both
//       5th-bit states (0/1) in each half — 64 combined.
//
// ANTI-HOLLOW (three arms 真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one decoded value) => ours-vs-oracle RED
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)                => ours-vs-oracle RED ONLY
//   -DINJECT=3  leaf nibble byte offset `+ 8` flipped to `+ 7` (harness cmp -l == 1)
//                                                                 => ours-vs-oracle RED ONLY
//
// argv: <nb(blocks·k=nb*32)> <reps> <seed> [flush_mb]
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

#define QK5_1 32
typedef _Float16 ggml_half;

// block_q5_1 = { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[QK5_1/2]; }
//            = 2 + 2 + 4 + 16 = 24.
static const int BLK = 24;

extern "C" void weft_emitc_dequant_q5_1_kernel_dequant_q5_1(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_q5_1(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_q5_1_kernel_dequant_q5_1
#define REF  dequantize_row_q5_1

// INDEPENDENT ZERO-MODEL oracle — re-decode q5_1 from raw bytes:
//   memcpy(qh, xb+4, 4); xh0=((qh>>j)<<4)&0x10; xh1=((qh>>(j+12)))&0x10;
//   x0 = (qs[j]&0x0F)|xh0; x1 = (qs[j]>>4)|xh1; y = fma(x,d,m).  (single fused rounding.)
static std::set<int> cov;   // (pos<<5)|(bit5<<4)|nibble  -> want all 64
static void oracle_q5_1(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK5_1;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d = (float)*(const _Float16*)(xb);       // fp16 block scale
        float m = (float)*(const _Float16*)(xb + 2);   // fp16 block min
        uint32_t qh; memcpy(&qh, xb + 4, sizeof(qh));  // 5th-bit plane
        const uint8_t* qs = xb + 8;                    // 16 packed nibble bytes
        float* yy = y + (size_t)i * QK5_1;
        for (int j = 0; j < QK5_1/2; ++j) {
            const uint8_t xh0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = (qs[j] & 0x0F) | xh0;
            int x1 = (qs[j] >>   4) | xh1;
            cov.insert((0<<5)|((xh0?1:0)<<4)|(qs[j]&0x0F));
            cov.insert((1<<5)|((xh1?1:0)<<4)|(qs[j]>>4));
            float mm = m;
#if INJECT==1
            int rx0 = (i==0 && j==0) ? x0 + 1 : x0;    // ORACLE-FAULT
            yy[j] = std::fma((float)rx0, d, mm);
#else
            yy[j] = std::fma((float)x0, d, mm);
#endif
            yy[j + QK5_1/2] = std::fma((float)x1, d, mm);
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
    int64_t k = (int64_t)nb * QK5_1;
    printf("=== q5_1 dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;   // d in [-0.5, 0.5]
        float mf = ((float)(rng() % 2001) - 1000.f) / 2000.f;   // m in [-0.5, 0.5]
        _Float16 dh = (_Float16)df; memcpy(xb, &dh, 2);
        _Float16 mh = (_Float16)mf; memcpy(xb + 2, &mh, 2);
        for (int b = 4; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff);  // qh + qs random
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_q5_1(blocks.data(), ora.data(), k);

    long mismOO = 0, mismRO = 0; int firstOO = -1;
    for (int64_t i = 0; i < k; ++i) {
        if (!(ours[i] == ora[i])) { if (firstOO<0) firstOO=(int)i; mismOO++; }
        if (!(ref[i]  == ora[i])) mismRO++;
    }
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n",
           mismOO, (long long)k, firstOO, mismOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s (FMA-contract consistency)\n",
           mismRO, (long long)k, mismRO==0?"PASS":"VOID-ORACLE");

    bool covok = (cov.size()==64);
    printf("  CORPUS nibble|5thbit=%zu/64 -> %s\n",
           cov.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

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
    printf("DEQUANT_ROW fmt=q5_1 nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
