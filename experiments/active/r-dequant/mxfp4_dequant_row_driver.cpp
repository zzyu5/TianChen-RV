// mxfp4_dequant_row_driver.cpp — R线 dequantize_row mxfp4 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_mxfp4 -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled OWNED real-vector leaf (per block: E8M0 block scale +
// vle8 the 16 packed nibble bytes + vand/vsrl nibble split + vrgather_vv_i8m1 through the
// register-resident 16-entry FP4 codebook + vsext_vf4 + vfcvt + vfmul_vf(d) + vse32 -- a
// REGISTER codebook gather, NOT a vluxei memory gather, so NO HW-gather wall). OPP/REF =
// ggml's deployed dequantize_row_mxfp4 (scalar-source host-autovec = the codegen-lottery
// opponent, 标量类档 per 对手法 §〇.1). Byte-exact ORACLE = an INDEPENDENT ZERO-MODEL
// recompute from the raw quantized bytes ([K-5]): mxfp4 decodes as d = GGML_E8M0_TO_FP32_HALF
// (e@0) via the EXACT bit construction (e<2 ? 0x00200000u<<e : (e-1)<<23, reinterpret as
// float), then y[j] = kvalues_mxfp4[qs[j]&0xF]*d (low 16) / kvalues_mxfp4[qs[j]>>4]*d
// (high 16). The fold is a SINGLE f32 multiply (no add/min => no fp-contraction ambiguity),
// so a correct real-vector emit is byte-exact by construction and the board `==` compare is
// the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives the E8M0 scale
//       + codebook itself; the DUT is the compiled RISC-V leaf. NOTHING is shared.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE; DUT, OPP and oracle
//       are each a pure function of those SAME bytes. Zero second stream.
//   (1) CORPUS COMPLETENESS: random bytes over nb blocks sweep every low nibble AND every
//       high nibble value (0..15 each, 32 combined); a counter prints + a gate refuses short.
//
// ANTI-HOLLOW (proves the byte-exact gate is not hollow · three arms 真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one decoded value) => ours-vs-oracle RED
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)                => ours-vs-oracle RED ONLY
//   -DINJECT=3  no driver change; linked against a leaf whose FP4 codebook byte `8, 12,`
//               was flipped to `8, 13,` (harness proves `cmp -l | wc -l` == 1)
//                                                                 => ours-vs-oracle RED ONLY
//               (the arm an x86 model pass cannot have: it bites the EMITTED RISC-V)
//
// argv: <nb(blocks·k=nb*32)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_mxfp4), flush each region
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

#define QK 32
// block_mxfp4 = { uint8_t e; uint8_t qs[16]; } = 17 bytes.
static const int BLK = 17;

extern "C" void weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_mxfp4(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4
#define REF  dequantize_row_mxfp4

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode mxfp4 from raw bytes (byte-exact to ggml's
// reference dequantize_row_mxfp4; the fold is a SINGLE f32 mul kv*d).
// ===========================================================================
static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3,  4,  6,  8,  12,
                                          0, -1, -2, -3, -4, -6, -8, -12};
static float e8m0_half(uint8_t e) {
    uint32_t bits = (e < 2) ? (0x00200000u << (e & 0x1F))
                            : (((uint32_t)e - 1u) << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
static std::set<int> cov;   // (pos<<4)|nibble  -> want all 32
static void oracle_mxfp4(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d = e8m0_half(xb[0]);
        const uint8_t* qs = xb + 1;
        float* yy = y + (size_t)i * QK;
        for (int j = 0; j < 16; ++j) {
            int nlo = qs[j] & 0xF; cov.insert((0 << 4) | nlo);
            float v = (float)kvalues_mxfp4[nlo] * d;
#if INJECT==1
            if (i == 0 && j == 0) v += 1.0f;   // ORACLE-FAULT
#endif
            yy[j] = v;
        }
        for (int j = 0; j < 16; ++j) {
            int nhi = qs[j] >> 4; cov.insert((1 << 4) | nhi);
            yy[j + 16] = (float)kvalues_mxfp4[nhi] * d;
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
    printf("=== mxfp4 dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    // ---------------- single source of truth: raw block bytes -------------------
    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        xb[0] = (uint8_t)(120 + (rng() % 17));           // E8M0 exponent -> d in [2^-8, 2^8]
        for (int b = 1; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff); // qs random nibbles
    }

    // ---------------- derived outputs (pure functions of the SAME bytes) --------
    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_mxfp4(blocks.data(), ora.data(), k);

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
    bool covok = (cov.size()==32);
    printf("  CORPUS nibble=%zu/32 -> %s\n", cov.size(), covok?"COMPLETE":"INCOMPLETE-CORPUS");

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
    printf("DEQUANT_ROW fmt=mxfp4 nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
