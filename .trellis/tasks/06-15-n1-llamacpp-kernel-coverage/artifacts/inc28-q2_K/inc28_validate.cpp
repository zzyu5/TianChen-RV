// INC-28 (q2_K) byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new typed op
// tcrv_rvv.q2_k_q8_k_block_dot computes the SAME fp32 dot-product output *s as
// ggml's own reference ggml_vec_dot_q2_K_q8_K_generic (llama.cpp/ggml/src/
// ggml-cpu/quants.c:514-564), BYTE-EXACT (exact IEEE-754 bit equality on the
// fp32 *s), over the full n set + edge cases incl. a MIN-term negative control
// (proving the -dmin*summs term is load-bearing).
//
// ORACLE = the VERBATIM ggml _generic function body (copied below, NOT a
// re-derivation). The one external macro GGML_CPU_FP16_TO_FP32 is shimmed to a
// hardware _Float16 cast (exact for every fp16: fp16->fp32 widening never
// rounds). q2_K = the 2-bit modern K-quant: 16 sub-blocks of 16, 2-bit weights
// (4 per qs byte), 4-bit scale (low nibble) + 4-bit min (high nibble) of the 16
// direct scales[16] bytes, scalar fp32 fold `sumf += dall*isum - dmin*summs`.
//
// Block byte layouts (ggml-common.h, QK_K = 256):
//   block_q2_K (84 bytes): scales[16]@0 | qs[64]@16 | d(fp16)@80 | dmin(fp16)@82
//   block_q8_K (292 bytes): d(fp32)@0 | qs[256]@4 | bsums[16]@260
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.
// Headline byte-exact regime: BOTH oracle and kernel compiled -ffp-contract=off.
// The kernel's fold `sumf += dall*isum - dmin*summs` is ONE emitc.expression with
// the SAME tree (two products, a subtract, an add) as _generic's single statement
// (quants.c:561), so it is contraction-stable -- under -ffp-contract=on/fast both
// sides may fuse `dall*isum` into the add, but they fuse IDENTICALLY (same source
// expression shape), so q2_K is mode-stable. Reported across off/on/fast.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#define QK_K 256

typedef uint16_t ggml_half;

typedef struct {
    uint8_t scales[QK_K/16]; // 16 packed 4-bit-scale(low)/4-bit-min(high) bytes
    uint8_t qs[QK_K/4];      // 64 packed 2-bit-weight bytes (4 per byte)
    union { struct { ggml_half d; ggml_half dmin; }; uint32_t dm; };
} block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

// ---- The one external macro: exact fp16->fp32 widening (no rounding). ----------
static inline float ggml_fp16_to_fp32_shim(uint16_t h) {
    _Float16 v;
    std::memcpy(&v, &h, sizeof(v));
    return (float)v;
}
#define GGML_CPU_FP16_TO_FP32(x) ggml_fp16_to_fp32_shim(x)
#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)

// ---- ggml's own reference, VERBATIM (quants.c:514-564). ------------------------
static void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s,
                                           size_t bs, const void * GGML_RESTRICT vx,
                                           size_t bx, const void * GGML_RESTRICT vy,
                                           size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = (const block_q2_K *) vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

// ---- The kernel our compiler emitted -------------------------------------------
// void f(size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
extern "C" void
tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

// A NEGATIVE-CONTROL variant that perturbs the MIN term (dmin -> dmin+1). It MUST
// mismatch the kernel whenever the min term (summs) is nonzero -- proving the
// -dmin*summs subtraction is genuinely load-bearing.
static void ggml_vec_dot_q2_K_q8_K_min_perturbed(int n, float *s, const void *vx,
                                                 const void *vy) {
    const block_q2_K *x = (const block_q2_K *) vx;
    const block_q8_K *y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * (GGML_CPU_FP16_TO_FP32(x[i].dmin) + 1.0f);
        int isum = 0; int is = 0; int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2; q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

// A NEGATIVE-CONTROL variant that drops the 2-bit MASK (`& 3`): it reads the full
// shifted byte instead of just the 2 low bits. It MUST mismatch the kernel
// whenever any high 2-bit field is set -- proving the 2-bit unpack mask is
// load-bearing (the q2_K-distinguishing weight decode).
static void ggml_vec_dot_q2_K_q8_K_no_mask(int n, float *s, const void *vx,
                                           const void *vy) {
    const block_q2_K *x = (const block_q2_K *) vx;
    const block_q8_K *y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
        int isum = 0; int is = 0; int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * (q2[l] >> shift); // NO & 3
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * (q2[l] >> shift); // NO & 3
                isum += d * isuml;
                shift += 2; q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

static inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline bool bit_equal(float a, float b) { return bits(a) == bits(b); }

static unsigned g_rng = 0x2c4f1b67u;
static unsigned next_rand() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

// fp16 bit pattern for a small "delta" magnitude (random, plausible scales).
static uint16_t random_fp16_delta() {
    uint16_t mant = (uint16_t)(next_rand() & 0x3FF);
    uint16_t exp  = (uint16_t)(8 + (next_rand() % 12)); // ~2^-7 .. 2^4
    return (uint16_t)((exp << 10) | mant);
}

static void fill_random_block(uint8_t *q2k, uint8_t *q8k) {
    for (int i = 0; i < 16; ++i) q2k[0 + i]  = (uint8_t)(next_rand() & 0xFF); // scales[16]
    for (int i = 0; i < 64; ++i) q2k[16 + i] = (uint8_t)(next_rand() & 0xFF); // qs[64]
    uint16_t d = random_fp16_delta();
    std::memcpy(q2k + 80, &d, 2);                       // fp16 d
    uint16_t dmin = random_fp16_delta();
    std::memcpy(q2k + 82, &dmin, 2);                    // fp16 dmin
    float dy = (float)((int)(next_rand() % 2000) - 1000) / 4096.0f; // fp32 activation d
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)(next_rand() & 0xFF); // qs
    for (int i = 0; i < 16; ++i) {
        int16_t bs = (int16_t)((int)(next_rand() % 4000) - 2000);
        std::memcpy(q8k + 260 + 2 * i, &bs, 2);          // bsums[16]
    }
}

// Compare *s for one buffer of NB super-blocks; returns 1 on bit-mismatch.
static int check_n(int NB, const char *label, int verbose) {
    size_t n = (size_t)NB * 256;
    uint8_t *vx = (uint8_t *)malloc((size_t)NB * 84);
    uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
    for (int ib = 0; ib < NB; ++ib)
        fill_random_block(vx + (size_t)ib * 84, vy + (size_t)ib * 292);

    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q2_K_q8_K_generic((int)n, &ref, 0, vx, 0, vy, 0, 1);
    tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(n, &got, vx, vy);

    int fail = !bit_equal(ref, got);
    if (fail || verbose)
        printf("  [%s] n=%zu  ref=%.9g (0x%08x)  tcrv=%.9g (0x%08x)  %s\n",
               label, n, ref, bits(ref), got, bits(got),
               fail ? "MISMATCH" : "bit-exact");
    free(vx); free(vy);
    return fail;
}

// Edge-case block builders (uniform fields for a controlled extreme).
static void build_edge(uint8_t *q2k, uint8_t *q8k, uint8_t qsByte,
                       uint8_t scaleByte, int8_t q8v, int16_t bsumv,
                       uint16_t dfp16, uint16_t dminfp16, float dy) {
    for (int i = 0; i < 16; ++i)  q2k[0 + i]  = scaleByte;
    for (int i = 0; i < 64; ++i)  q2k[16 + i] = qsByte;
    std::memcpy(q2k + 80, &dfp16, 2);
    std::memcpy(q2k + 82, &dminfp16, 2);
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)q8v;
    for (int i = 0; i < 16; ++i) std::memcpy(q8k + 260 + 2 * i, &bsumv, 2);
}

static int check_edge(const char *label, uint8_t qsByte, uint8_t scaleByte,
                      int8_t q8v, int16_t bsumv, uint16_t dfp16,
                      uint16_t dminfp16, float dy) {
    uint8_t q2k[84]; uint8_t q8k[292];
    build_edge(q2k, q8k, qsByte, scaleByte, q8v, bsumv, dfp16, dminfp16, dy);
    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q2_K_q8_K_generic(256, &ref, 0, q2k, 0, q8k, 0, 1);
    tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(256, &got, q2k, q8k);
    int fail = !bit_equal(ref, got);
    printf("  [edge %s] ref=%.9g (0x%08x) tcrv=%.9g (0x%08x) %s\n",
           label, ref, bits(ref), got, bits(got),
           fail ? "MISMATCH" : "bit-exact");
    return fail;
}

int main() {
    int failures = 0;
    int checked = 0;

    printf("INC-28 q2_K byte-exact *s vs ggml _generic (oracle = verbatim _generic)\n");

    // ---- The named n set ---------------------------------------------------------
    printf("named n set:\n");
    const int ns[] = {1, 2, 8, 16, 100, 256}; // *256 = {256,512,2048,4096,25600,65536}
    const char *nlabels[] = {"n=256","n=512","n=2048","n=4096","n=25600","n=65536"};
    for (int i = 0; i < 6; ++i) {
        failures += check_n(ns[i], nlabels[i], 1); ++checked;
    }

    // ---- Edge cases on extremes --------------------------------------------------
    printf("edge cases:\n");
    // d magnitudes: fp16 1.0=0x3C00, 0.5=0x3800, 2.0=0x4000, 1.5=0x3E00.
    // all-zero: q2=0, scale=0 (sc&0xF=0, sc>>4=0), q8=0, bsum=0 -> *s = 0.
    failures += check_edge("all-zero (q2=0 scale=0 q8=0 bsum=0 d=dmin=1.0 dy=1.0)",
                           0x00, 0x00, 0, 0, 0x3C00, 0x3C00, 1.0f); ++checked;
    // q2 = 3 EVERYWHERE: qs=0xFF -> every 2-bit field is 3. scale=0xFF: sc&0xF=15,
    // sc>>4=15 (both nibble extremes). Exercises the full positive AND min term.
    failures += check_edge("q2=3 (qs=0xFF) scale=0xFF (sc=15,min=15) q8=+127 bsum=+2000 d=2.0 dmin=0.5 dy=-3.5",
                           0xFF, 0xFF, 127, 2000, 0x4000, 0x3800, -3.5f); ++checked;
    // q2 = 0 EVERYWHERE (qs=0): isum=0 -> *s = -dmin*summs (the MIN term ALONE).
    failures += check_edge("min-only: q2=0 (qs=0 -> isum=0) scale=0xF0 (sc=0,min=15) q8=+1 bsum=+1500 dmin=2.0",
                           0x00, 0xF0, 1, 1500, 0x3C00, 0x4000, 1.5f); ++checked;
    // scale-only: min nibble 0 (scale=0x0F: sc&0xF=15, sc>>4=0) -> summs=0 -> only
    // the positive term. q2=2 (qs=0xAA -> each field is 2).
    failures += check_edge("scale-only: scale=0x0F (sc=15,min=0 -> summs=0) q2=2 (qs=0xAA) q8=-128 bsum=-2000 d=0.5 dy=2.25",
                           0xAA, 0x0F, -128, -2000, 0x3800, 0x4000, 2.25f); ++checked;
    // q2 = 1 EVERYWHERE (qs=0x55 -> each 2-bit field is 1): the low-bit plane.
    failures += check_edge("q2=1 (qs=0x55) scale=0x5A (sc=10,min=5) q8=+3 bsum=+500 d=1.5 dmin=0.5 dy=1.0",
                           0x55, 0x5A, 3, 500, 0x3E00, 0x3800, 1.0f); ++checked;
    // mixed qs=0x1B (bit fields 3,1,2,0 across the 4 shifts): exercises each shift.
    failures += check_edge("qs=0x1B (fields 3,2,1,0 by shift) scale=0xA5 (sc=5,min=10) q8=-3 bsum=-500 d=0.5 dmin=1.5 dy=1.0",
                           0x1B, 0xA5, -3, -500, 0x3800, 0x3E00, 1.0f); ++checked;
    // dy = 0 -> *s must be exactly 0 (both dall and dmin vanish).
    failures += check_edge("dy=0.0 -> *s must be 0 (both terms vanish)",
                           0xAB, 0x55, 33, 777, 0x3C00, 0x3C00, 0.0f); ++checked;

    // ---- Many random n -----------------------------------------------------------
    printf("random n (NB in 1..32), 2000 trials:\n");
    int random_fail = 0;
    for (int t = 0; t < 2000; ++t) {
        int NB = 1 + (int)(next_rand() % 32);
        char label[24]; snprintf(label, sizeof(label), "rnd#%d NB=%d", t, NB);
        int f = check_n(NB, label, 0);
        random_fail += f; ++checked;
    }
    failures += random_fail;
    printf("  random trials with a mismatch: %d / 2000\n", random_fail);

    // ---- Negative control 1: MIN-perturbed oracle MUST mismatch the kernel -------
    printf("MIN-term negative control (perturbed dmin must MISMATCH):\n");
    int neg_discriminating = 0, neg_total = 0;
    for (int t = 0; t < 200; ++t) {
        int NB = 1 + (int)(next_rand() % 8);
        size_t n = (size_t)NB * 256;
        uint8_t *vx = (uint8_t *)malloc((size_t)NB * 84);
        uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
        for (int ib = 0; ib < NB; ++ib)
            fill_random_block(vx + (size_t)ib * 84, vy + (size_t)ib * 292);
        float bad = 0.0f, got = 0.0f;
        ggml_vec_dot_q2_K_q8_K_min_perturbed((int)n, &bad, vx, vy);
        tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(n, &got, vx, vy);
        if (!bit_equal(bad, got)) ++neg_discriminating;
        ++neg_total;
        free(vx); free(vy);
    }
    printf("  min-perturbed-vs-kernel mismatched: %d / %d (must be %d)\n",
           neg_discriminating, neg_total, neg_total);
    if (neg_discriminating != neg_total) {
        printf("  NEGATIVE CONTROL FAILED: harness does not discriminate the MIN term!\n");
        ++failures;
    }

    // ---- Negative control 2: the 2-bit-mask-DROPPING oracle MUST mismatch --------
    // the kernel whenever any high 2-bit field is set -- proving the `& 3` weight
    // unpack mask is genuinely load-bearing (the q2_K-distinguishing weight decode).
    printf("2-bit-unpack negative control (no-mask `>> shift` w/o `& 3` must MISMATCH):\n");
    int mask_discriminating = 0, mask_total = 0;
    for (int t = 0; t < 200; ++t) {
        int NB = 1 + (int)(next_rand() % 8);
        size_t n = (size_t)NB * 256;
        uint8_t *vx = (uint8_t *)malloc((size_t)NB * 84);
        uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
        for (int ib = 0; ib < NB; ++ib) {
            uint8_t *bx = vx + (size_t)ib * 84;
            fill_random_block(bx, vy + (size_t)ib * 292);
            // Force every qs byte's high fields nonzero so the no-mask divergence
            // is guaranteed (bit 7 or 6 set -> `>>shift` leaks into low fields).
            for (int i = 0; i < 64; ++i) bx[16 + i] = (uint8_t)(next_rand() | 0xC0);
        }
        float nomask = 0.0f, got = 0.0f;
        ggml_vec_dot_q2_K_q8_K_no_mask((int)n, &nomask, vx, vy);
        tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(n, &got, vx, vy);
        if (!bit_equal(nomask, got)) ++mask_discriminating;
        ++mask_total;
        free(vx); free(vy);
    }
    printf("  no-mask-vs-kernel mismatched: %d / %d (must be %d)\n",
           mask_discriminating, mask_total, mask_total);
    if (mask_discriminating != mask_total) {
        printf("  2-bit NEGATIVE CONTROL FAILED: kernel ignores the unpack mask!\n");
        ++failures;
    }

    printf("\nINC-28 q2_K: %d positive cases checked, %d failures\n",
           checked, failures);
    if (failures == 0)
        printf("RESULT: PASS (*s bit-exact vs ggml _generic for all cases; "
               "MIN-term + 2-bit-unpack negative controls discriminating)\n");
    else
        printf("RESULT: FAIL\n");
    return failures == 0 ? 0 : 1;
}
