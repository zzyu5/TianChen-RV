// nvfp4_dequant_row_driver.cpp — R线 dequantize_row nvfp4 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_nvfp4 -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled OWNED real-vector leaf (per 16-element sub-block s: a
// UE4M3 sub-scale d[s] + vle8 the 8 packed nibble bytes + vand/vsrl nibble split +
// vrgather_vv_i8m1 through the register-resident 16-entry FP4 codebook + vsext_vf4 + vfcvt +
// vfmul_vf(d[s]) + vse32 -- a REGISTER codebook gather, NOT a vluxei memory gather, so NO
// HW-gather wall). OPP/REF = ggml's deployed dequantize_row_nvfp4 (scalar-source host-autovec
// = the codegen-lottery opponent, 标量类档 per 对手法 §〇.1). Byte-exact ORACLE = an
// INDEPENDENT ZERO-MODEL recompute from the raw quantized bytes ([K-5]): nvfp4 decodes as four
// UE4M3 sub-scales d[s] = ggml_ue4m3_to_fp32(x.d[s]) (ldexpf-based), then per sub s the 8
// nibble bytes at qs+s*8 give yb[j] = kvalues_mxfp4[qs&0xF]*d[s] (low 8) /
// kvalues_mxfp4[qs>>4]*d[s] (high 8) into yb = y + s*16. The fold is a SINGLE f32 multiply (no
// add/min => no fp-contraction ambiguity), so a correct real-vector emit is byte-exact by
// construction and the board `==` compare is the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives the UE4M3 sub
//       scales + codebook itself; the DUT is the compiled RISC-V leaf. NOTHING is shared.
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
// argv: <nb(blocks·k=nb*64)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_nvfp4), flush each region
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

#define QK 64
#define QK_SUB 16
#define N_SUB 4
// block_nvfp4 = { uint8_t d[4]; uint8_t qs[32]; } = 36 bytes.
static const int BLK = 36;

extern "C" void weft_emitc_dequant_nvfp4_kernel_dequant_nvfp4(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_nvfp4(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_nvfp4_kernel_dequant_nvfp4
#define REF  dequantize_row_nvfp4

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode nvfp4 from raw bytes (byte-exact to ggml's
// reference dequantize_row_nvfp4; the fold is a SINGLE f32 mul kv*d[s]).
// ggml_ue4m3_to_fp32 (ggml-impl.h): e==0||e==0x7F -> 0; exp=(e>>3)&0xF, man=e&7;
// raw = exp==0 ? ldexpf(man,-9) : ldexpf(1+man/8, exp-7); result = raw*0.5f.
// ===========================================================================
static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3,  4,  6,  8,  12,
                                          0, -1, -2, -3, -4, -6, -8, -12};
static float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw;
    if (exp == 0) raw = ldexpf((float)man, -9);
    else          raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}
static std::set<int> cov;   // (pos<<4)|nibble  -> want all 32
static void oracle_nvfp4(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        for (int s = 0; s < N_SUB; ++s) {
            float d = ue4m3_to_fp32(xb[s]);
            float* yb = y + (size_t)i * QK + s * QK_SUB;
            const uint8_t* qs = xb + 4 + s * (QK_SUB / 2);
            for (int j = 0; j < QK_SUB / 2; ++j) {
                int nlo = qs[j] & 0xF; cov.insert((0 << 4) | nlo);
                float v = (float)kvalues_mxfp4[nlo] * d;
#if INJECT==1
                if (i == 0 && s == 0 && j == 0) v += 1.0f;   // ORACLE-FAULT
#endif
                yb[j] = v;
                int nhi = qs[j] >> 4; cov.insert((1 << 4) | nhi);
                yb[j + QK_SUB / 2] = (float)kvalues_mxfp4[nhi] * d;
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
    int64_t k = (int64_t)nb * QK;
    printf("=== nvfp4 dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        for (int s = 0; s < N_SUB; ++s)
            xb[s] = (uint8_t)(1 + (rng() % 0x7E));           // UE4M3 in [1,0x7E]: finite non-zero
        for (int b = 4; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff); // qs random nibbles
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_nvfp4(blocks.data(), ora.data(), k);

    long mismOO = 0, mismRO = 0; int firstOO = -1;
    for (int64_t i = 0; i < k; ++i) {
        if (!(ours[i] == ora[i])) { if (firstOO<0) firstOO=(int)i; mismOO++; }
        if (!(ref[i]  == ora[i])) mismRO++;
    }
    printf("  BYTE-EXACT ours-vs-oracle: mism=%ld/%lld first=%d -> %s\n",
           mismOO, (long long)k, firstOO, mismOO==0?"PASS":"VOID-CORRECTNESS");
    printf("  CROSSCHECK ggml-vs-oracle: mism=%ld/%lld -> %s\n",
           mismRO, (long long)k, mismRO==0?"PASS":"VOID-ORACLE");

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
    printf("DEQUANT_ROW fmt=nvfp4 nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
