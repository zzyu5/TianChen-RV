// tq2_0_dequant_row_driver.cpp — R线 dequantize_row tq2_0 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_tq2_0 -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled OWNED real-vector leaf (per super-block: fp16 d + vle8 the
// 2-bit-packed qs bytes + vsrl_vx/vand_vx the 2-bit ternary field + reinterpret u8->i8 +
// vsext_vf4 + vsub_vx(1) + vfcvt + vfmul_vf(d) + vse32 -- PURE ARITHMETIC ternary decode, NO
// codebook and NO gather). OPP/REF = ggml's deployed dequantize_row_tq2_0 (scalar-source
// host-autovec = the codegen-lottery opponent, 标量类档 per 对手法 §〇.1). Byte-exact ORACLE =
// an INDEPENDENT ZERO-MODEL recompute from the raw quantized bytes ([K-5]): tq2_0 decodes as
// d = fp16@64, then over j in {0,32}, l in 0..3, m in 0..31: q = (qs[j+m] >> (2l)) & 3,
// y[out++] = (q-1)*d. The fold is a SINGLE f32 multiply (no add/min => no fp-contraction
// ambiguity), so a correct real-vector emit is byte-exact by construction and the board `==`
// compare is the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives d + the 2-bit
//       ternary decode itself; the DUT is the compiled RISC-V leaf. NOTHING is shared.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE; DUT, OPP and oracle
//       are each a pure function of those SAME bytes. Zero second stream.
//   (1) CORPUS COMPLETENESS: random bytes over nb blocks sweep every 2-bit value (0..3) at
//       every l-position (0..3) => 16 combos; a counter prints + a gate refuses short.
//
// ANTI-HOLLOW (proves the byte-exact gate is not hollow · three arms 真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one decoded value) => ours-vs-oracle RED
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)                => ours-vs-oracle RED ONLY
//   -DINJECT=3  no driver change; linked against a leaf whose fp16 d-read offset `v8 + 64;`
//               was shifted to `v8 + 63;` (harness proves `cmp -l | wc -l` == 1)
//                                                                 => ours-vs-oracle RED ONLY
//               (the arm an x86 model pass cannot have: it bites the EMITTED RISC-V)
//
// argv: <nb(blocks·k=nb*256)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_tq2_0), flush each region
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

#define QK 256
// block_tq2_0 = { uint8_t qs[64]; ggml_half d; } = 66 bytes.
static const int BLK = 66;

extern "C" void weft_emitc_dequant_tq2_0_kernel_dequant_tq2_0(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_tq2_0(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_tq2_0_kernel_dequant_tq2_0
#define REF  dequantize_row_tq2_0

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode tq2_0 from raw bytes (byte-exact to ggml's
// reference dequantize_row_tq2_0; the fold is a SINGLE f32 mul (q-1)*d).
// ===========================================================================
static std::set<int> cov;   // (l<<2)|q  -> want all 16
static void oracle_tq2_0(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        const uint8_t* qs = xb;                     // qs[64] @0
        float d = (float)*(const _Float16*)(xb + 64);
        float* yy = y + (size_t)i * QK;
        long out = 0;
        for (int j = 0; j < 64; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 32; ++m) {
                    int q = (qs[j + m] >> (l * 2)) & 3;
                    cov.insert((l << 2) | q);
                    float v = (float)(q - 1) * d;
#if INJECT==1
                    if (i == 0 && out == 0) v += 1.0f;   // ORACLE-FAULT
#endif
                    yy[out++] = v;
                }
            }
        }
    }
}

// ===========================================================================
// timing helpers (cold · flush > LLC before each timed region)
// ===========================================================================
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
    int64_t k = (int64_t)nb * QK;
    printf("=== tq2_0 dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    // ---------------- single source of truth: raw block bytes -------------------
    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        for (int b = 0; b < 64; ++b) xb[b] = (uint8_t)(rng() & 0xff);  // qs random 2-bit fields
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;         // d in [-0.5, 0.5]
        _Float16 dh = (_Float16)df; memcpy(xb + 64, &dh, 2);
    }

    // ---------------- derived outputs (pure functions of the SAME bytes) --------
    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_tq2_0(blocks.data(), ora.data(), k);

    // ---------------- BYTE-EXACT (f32 `==`, ZERO-MODEL, decode-only) ------------
    long mismOO = 0, mismRO = 0; int firstOO = -1;
    for (int64_t i = 0; i < k; ++i) {
        if (!(ours[i] == ora[i])) { if (firstOO<0) firstOO=(int)i; mismOO++; }
        if (!(ref[i]  == ora[i])) mismRO++;
    }
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n",
           mismOO, (long long)k, firstOO, mismOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n",
           mismRO, (long long)k, mismRO==0?"PASS":"VOID-ORACLE");

    // ---------------- corpus completeness ---------------------------------------
    bool covok = (cov.size()==16);
    printf("  CORPUS ternary=%zu/16 -> %s\n", cov.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

    if (reps == 0) {
        bool ok = (mismOO==0 && mismRO==0);
        if (INJECT==0 && !covok) { fprintf(stderr,"INCOMPLETE-CORPUS: raise nb\n"); return 9; }
        return ok ? 0 : 1;
    }

    if (mismOO != 0) {
        fprintf(stderr,"VOID-CORRECTNESS: ours mismatched oracle => timing forbidden\n");
        return 1;
    }

    // ---------------- COLD paired ours/ggml, flush before EACH timed region -----
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
    printf("DEQUANT_ROW fmt=tq2_0 nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
