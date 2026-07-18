// q8_0_dequant_row_driver.cpp — R线 dequantize_row q8_0 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_q8_0 -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled emitted leaf (OWNED real-vector: vle8 + vsext_vf4 +
// vfcvt + vfmul_vf + vse32); OPP/REF = ggml's deployed dequantize_row_q8_0 (scalar-source,
// host-autovec = the codegen-lottery opponent, 标量类档 per 对手法 §〇.1). Byte-exact
// ORACLE = an INDEPENDENT ZERO-MODEL recompute from the raw quantized bytes ([K-5]): q8_0
// dequant is `y[j] = qs[j] * d` — the ONLY rounding is the single f32 multiply (NO add /
// NO min => NO fp-contraction ambiguity), so a correct real-vector emit is byte-exact to
// the scalar reference by construction, and the board `==` compare is the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives d/qs itself;
//       the DUT is the compiled RISC-V leaf. NOTHING is shared (q8_0 has no codebook /
//       table). A wrong decode is CAUGHT.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE; DUT, OPP and
//       oracle every one is a pure function of those SAME bytes. Zero second stream.
//   (1) CORPUS COMPLETENESS: random bytes over nb blocks sweep every signed-int8 quant
//       value (-128..127, all 256); a counter prints + a gate refuses a short corpus.
//
// ANTI-HOLLOW (proves the byte-exact gate is not hollow · three arms真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one quant value) => ours-vs-oracle RED
//                              (proves the oracle actually recomputes, does not echo ours)
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)             => ours-vs-oracle RED ONLY
//                              (proves the compare bites the DUT output)
//   -DINJECT=3  no driver change; linked against a leaf whose quant byte offset `+ 2` was
//               flipped to `+ 1` (harness proves `cmp -l | wc -l` == 1)
//                                                              => ours-vs-oracle RED ONLY
//               (the arm an x86 model pass cannot have: it bites the EMITTED RISC-V)
//
// argv: <nb(blocks·k=nb*32)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_q8_0), flush each region
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

#define QK8_0 32
typedef _Float16 ggml_half;

// block_q8_0 = { ggml_half d; int8_t qs[QK8_0]; } = 2 + 32 = 34 bytes.
static const int BLK = 34;

// ===========================================================================
// DUT (our emitted leaf) + OPP/REF (ggml deployed dequantize_row)
// ===========================================================================
extern "C" void weft_emitc_dequant_q8_0_kernel_dequant_q8_0(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_q8_0(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_q8_0_kernel_dequant_q8_0
#define REF  dequantize_row_q8_0

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode q8_0 from raw bytes: y[j] = qs[j]*d
// (single f32 rounding, no reassociation). INJECT=1 rotates one quant value so
// ours-vs-oracle must go RED (anti-hollow arm A).
// ===========================================================================
static std::set<int> cov_q;
static void oracle_q8_0(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d = (float)*(const _Float16*)(xb);          // fp16 block scale
        const int8_t* qs = (const int8_t*)(xb + 2);       // 32 signed int8 quants
        float* yy = y + (size_t)i * QK8_0;
        for (int j = 0; j < QK8_0; ++j) {
            int q = qs[j];
#if INJECT==1
            if (i == 0 && j == 0) q = (int8_t)(q + 1);     // ORACLE-FAULT
#endif
            cov_q.insert(q & 0xff);
            yy[j] = (float)q * d;
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
    int64_t k = (int64_t)nb * QK8_0;
    printf("=== q8_0 dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    // ---------------- single source of truth: raw block bytes -------------------
    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        // d: a small random fp16 (decode is linear in d; magnitude irrelevant to exactness)
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;   // [-0.5, 0.5]
        _Float16 dh = (_Float16)df; memcpy(xb, &dh, 2);
        // qs: any int8 pattern is a valid q8_0 payload — random sweeps every quant value.
        for (int b = 2; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff);
    }

    // ---------------- derived outputs (pure functions of the SAME bytes) --------
    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_q8_0(blocks.data(), ora.data(), k);

    // ---------------- BYTE-EXACT (f32 `==`, ZERO-MODEL, decode-only) ------------
    // ours-vs-oracle = the DUT certificate (independent recompute).
    // ref-vs-oracle  = cross-check that ggml's deployed decode agrees with the oracle
    //                  (both must be bit-identical — decode is deterministic single-round).
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
    bool covok = (cov_q.size()==256);
    printf("  CORPUS quant_val=%zu/256 -> %s\n",
           cov_q.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

    if (reps == 0) {
        // verify-only: correctness must hold; corpus is advisory unless INJECT==0
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
    printf("DEQUANT_ROW fmt=q8_0 nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
