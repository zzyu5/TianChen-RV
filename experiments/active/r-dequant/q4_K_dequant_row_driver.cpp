// q4_K_dequant_row_driver.cpp — R线 dequantize_row q4_K 每格对拍/计时 driver (standalone leaf).
//
// op = dequantize_row (block_q4_K -> f32 row · void-return · NO reduction · NO opponent
// vec_dot). DUT = our board-compiled OWNED real-vector leaf (per super-sub: vle8 + vand/vsrl
// nibble split + vzext_vf4 + vfcvt + vfmv_v_f(m1)+vfmsac_vf(d1) FUSED d1*v - m1 + vse32);
// OPP/REF = ggml's deployed dequantize_row_q4_K (scalar-source host-autovec = the
// codegen-lottery opponent, 标量类档 per 对手法 §〇.1). Byte-exact ORACLE = an INDEPENDENT
// ZERO-MODEL recompute from the raw quantized bytes ([K-5]): q4_K decodes as d=fp16@0,
// dmin=fp16@2, get_scale_min_k4 6-bit (sc,m) pairs from scales[12]@4, then per super-sub jj
// y = d1*(qs&0xF) - m1 (low 32) / d2*(qs>>4) - m2 (high 32) with d1=d*sc, m1=dmin*m. The fold
// is a SINGLE fused mul-sub (std::fma) matching the -ffp-contract=on opponent autovec's
// vfmsub and our vfmsac -- so a correct real-vector emit is byte-exact by construction and
// the board `==` compare is the certificate.
//
// [K-5b] THREE REQUIREMENTS
//   (3) ORACLE INDEPENDENT: oracle reads the raw block bytes and re-derives d/dmin/scales/
//       nibbles itself; the DUT is the compiled RISC-V leaf. NOTHING is shared.
//   (2) INPUT PATH SAME-SOURCE: ONE rng fills the raw block bytes ONCE; DUT, OPP and oracle
//       are each a pure function of those SAME bytes. Zero second stream.
//   (1) CORPUS COMPLETENESS: random bytes over nb blocks sweep every low nibble AND every
//       high nibble value (0..15 each, 32 combined); a counter prints + a gate refuses short.
//
// ANTI-HOLLOW (proves the byte-exact gate is not hollow · three arms 真隔离):
//   -DINJECT=1  ORACLE-FAULT   (oracle rotates one decoded value) => ours-vs-oracle RED
//   -DINJECT=2  DUT-FAULT      (ours[mid] += 1.0f)                => ours-vs-oracle RED ONLY
//   -DINJECT=3  no driver change; linked against a leaf whose qs base offset `v8 + 16` was
//               flipped to `v8 + 17` (harness proves `cmp -l | wc -l` == 1)
//                                                                 => ours-vs-oracle RED ONLY
//               (the arm an x86 model pass cannot have: it bites the EMITTED RISC-V)
//
// argv: <nb(blocks·k=nb*256)> <reps> <seed> [flush_mb]
//   reps==0  => verify-only (byte-exact + corpus, NO timing)
//   reps>0   => cold N=reps timed (ours-leaf vs ggml dequantize_row_q4_K), flush each region
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

// block_q4_K = { fp16 d; fp16 dmin; uint8_t scales[12]; uint8_t qs[128]; } = 144 bytes.
static const int BLK = 144;

// ===========================================================================
// DUT (our emitted leaf) + OPP/REF (ggml deployed dequantize_row)
// ===========================================================================
extern "C" void weft_emitc_dequant_q4_K_kernel_dequant_q4_K(
    size_t k, const uint8_t* x, float* y);
extern "C" void dequantize_row_q4_K(const void* x, float* y, int64_t k);
#define LEAF weft_emitc_dequant_q4_K_kernel_dequant_q4_K
#define REF  dequantize_row_q4_K

// ===========================================================================
// INDEPENDENT ZERO-MODEL oracle — re-decode q4_K from raw bytes (byte-exact to ggml's
// reference dequantize_row_q4_K; the fold is the fused d1*v - m1 via std::fma).
// INJECT=1 rotates one decoded value so ours-vs-oracle must go RED (anti-hollow arm A).
// ===========================================================================
static std::set<int> cov;   // (pos<<4)|nibble  -> want all 32
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}
static void oracle_q4_K(const uint8_t* xall, float* y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* xb = xall + (size_t)i * BLK;
        float d  = (float)*(const _Float16*)(xb);
        float mn = (float)*(const _Float16*)(xb + 2);
        const uint8_t* scales = xb + 4;
        const uint8_t* q = xb + 16;
        float* yy = y + (size_t)i * QK_K;
        int is = 0;
        for (int seg = 0; seg < QK_K; seg += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) {
                int nlo = q[l] & 0xF; cov.insert((0 << 4) | nlo);
                float v = std::fma(d1, (float)nlo, -m1);
#if INJECT==1
                if (i == 0 && seg == 0 && l == 0) v += 1.0f;   // ORACLE-FAULT
#endif
                *yy++ = v;
            }
            for (int l = 0; l < 32; ++l) {
                int nhi = q[l] >> 4; cov.insert((1 << 4) | nhi);
                *yy++ = std::fma(d2, (float)nhi, -m2);
            }
            q += 32; is += 2;
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
    int64_t k = (int64_t)nb * QK_K;
    printf("=== q4_K dequantize_row | nb=%d k=%lld seed=0x%X INJECT=%d\n",
           nb, (long long)k, seed, INJECT);

    // ---------------- single source of truth: raw block bytes -------------------
    std::mt19937 rng(seed);
    std::vector<uint8_t> blocks((size_t)nb * BLK);
    for (int i = 0; i < nb; ++i) {
        uint8_t* xb = &blocks[(size_t)i * BLK];
        float df = ((float)(rng() % 2001) - 1000.f) / 2000.f;   // d in [-0.5, 0.5]
        float mf = ((float)(rng() % 2001) - 1000.f) / 4000.f;   // dmin
        _Float16 dh = (_Float16)df; memcpy(xb, &dh, 2);
        _Float16 mh = (_Float16)mf; memcpy(xb + 2, &mh, 2);
        for (int b = 4; b < BLK; ++b) xb[b] = (uint8_t)(rng() & 0xff); // scales + qs random
    }

    // ---------------- derived outputs (pure functions of the SAME bytes) --------
    std::vector<float> ours(k,0.f), ref(k,0.f), ora(k,0.f);
    LEAF((size_t)k, blocks.data(), ours.data());
#if INJECT==2
    ours[k/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    REF(blocks.data(), ref.data(), k);
    oracle_q4_K(blocks.data(), ora.data(), k);

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
    printf("DEQUANT_ROW fmt=q4_K nb=%d k=%lld reps=%d seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "ggml_med_ns=%.0f(iqr%.2f%%) | ratio_cold=%.4f sink=%llu\n",
           nb,(long long)k,reps,seed,rM,riqr(tR),gM,riqr(tG),gM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
