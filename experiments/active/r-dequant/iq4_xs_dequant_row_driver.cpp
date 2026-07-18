// iq4_xs_dequant_row_driver.cpp — R线 dequantize_row iq4_xs 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_iq4_xs -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled OWNED real-vector leaf (per 32-element sub-block ib: a
// super-block fp16 d + a signed-6 sub scale dl=d*(ls-32) from a scales_l nibble + a scales_h
// 2-bit + vle8 the 16 packed nibble bytes + vand/vsrl nibble split + vrgather_vv_i8m1 through
// the register-resident 16-entry non-linear codebook + vsext_vf4 + vfcvt + vfmul_vf(dl) +
// vse32 -- a REGISTER codebook gather, NOT a vluxei memory gather, so NO HW-gather wall).
// OPP/REF = ggml's deployed dequantize_row_iq4_xs (scalar-source host-autovec = the
// codegen-lottery opponent, 标量类档 per 对手法 §〇.1). Byte-exact ORACLE = an INDEPENDENT
// ZERO-MODEL recompute from the raw quantized bytes ([K-5]): iq4_xs decodes as d = fp16@0,
// scales_h = u16@2, scales_l[4]@4; per ib (8 sub-blocks of 32) ls = ((scales_l[ib/2] >>
// 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4), dl = d*(ls-32); then y[j] =
// dl*kvalues_iq4nl[qs[j]&0xf] (low 16) / dl*kvalues_iq4nl[qs[j]>>4] (high 16) with qs at +8.
// The scale seam dl and the codebook fold are BOTH single f32 multiplies (no add/min => no
// fp-contraction ambiguity), so a correct real-vector emit is byte-exact by construction and
// the board `==` compare is the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives d/scales_h/
//       scales_l/nibbles + codebook itself; the DUT is the compiled RISC-V leaf. NOTHING shared.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE; DUT, OPP and oracle
//       are each a pure function of those SAME bytes. Zero second stream.
//   (1) CORPUS COMPLETENESS: random bytes over nb blocks sweep every low nibble AND every
//       high nibble value (0..15 each, 32 combined); a counter prints + a gate refuses short.
//
// ANTI-HOLLOW (proves the byte-exact gate is not hollow · three arms 真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one decoded value) => ours-vs-oracle RED
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)                => ours-vs-oracle RED ONLY
//   -DINJECT=3  no driver change; linked against a leaf whose codebook byte `89, 113}` was
//               flipped to `89, 114}` (harness proves `cmp -l | wc -l` == 1)
//                                                                 => ours-vs-oracle RED ONLY
//               (the arm an x86 model pass cannot have: it bites the EMITTED RISC-V)
//
// argv: <nb(blocks·k=nb*256)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_iq4_xs), flush each region
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
// block_iq4_xs = { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } = 136 bytes.
static const int BLK = 136;

extern "C" void weft_emitc_dequant_iq4_xs_kernel_dequant_iq4_xs(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_iq4_xs(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_iq4_xs_kernel_dequant_iq4_xs
#define REF  dequantize_row_iq4_xs

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode iq4_xs from raw bytes (byte-exact to ggml's
// reference dequantize_row_iq4_xs; the scale dl and the codebook fold are SINGLE f32 muls).
// ===========================================================================
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                          1,    13,   25,  38,  53,  69,  89,  113};
static std::set<int> cov;   // (pos<<4)|nibble  -> want all 32
static void oracle_iq4_xs(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d = (float)*(const _Float16*)(xb);
        int scales_h = (int)xb[2] | ((int)xb[3] << 8);   // uint16 LE
        const uint8_t* scales_l = xb + 4;
        const uint8_t* qs = xb + 8;
        float* yy = y + (size_t)i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; ++j) {
                int nlo = qs[j] & 0xf; cov.insert((0 << 4) | nlo);
                float v = dl * (float)kvalues_iq4nl[nlo];
#if INJECT==1
                if (i == 0 && ib == 0 && j == 0) v += 1.0f;   // ORACLE-FAULT
#endif
                yy[j] = v;
                int nhi = qs[j] >> 4; cov.insert((1 << 4) | nhi);
                yy[j + 16] = dl * (float)kvalues_iq4nl[nhi];
            }
            yy += 32;
            qs += 16;
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
    printf("=== iq4_xs dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;   // d in [-0.5, 0.5]
        _Float16 dh = (_Float16)df; memcpy(xb, &dh, 2);
        for (int b = 2; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff); // scales_h/l + qs random
    }

    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_iq4_xs(blocks.data(), ora.data(), k);

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
    printf("DEQUANT_ROW fmt=iq4_xs nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
