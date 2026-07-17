// INC-24 (q4_K K4b) byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new typed op
// tcrv_rvv.q4_k_q8_k_block_dot computes the SAME fp32 dot-product output *s as
// ggml's own reference ggml_vec_dot_q4_K_q8_K_generic (llama.cpp/ggml/src/
// ggml-cpu/quants.c:645-718), BYTE-EXACT (exact IEEE-754 bit equality on the
// fp32 *s), over the full n set + edge cases incl. a MIN-term negative control.
//
// ORACLE = the VERBATIM ggml _generic function body (copied below, NOT a
// re-derivation). The one external macro GGML_CPU_FP16_TO_FP32 is shimmed to a
// hardware _Float16 cast (exact for every fp16: fp16->fp32 widening never
// rounds). K4b adds the d/dmin fp32 fold + the q4_K MIN term (Σ bsums*mins) +
// the ABI on top of K4a's integer core, so this reads NONZERO fp16 d/dmin AND
// nonzero q8_K bsums (unlike K4a, which exercised neither).
//
// Block byte layouts (ggml-common.h, QK_K = 256):
//   block_q4_K (144 bytes): d(fp16) @0 | dmin(fp16) @2 | scales[12] @4 | qs[128] @16
//   block_q8_K (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.
// Headline byte-exact regime: BOTH oracle and kernel compiled -ffp-contract=off
// (the kernel's positive fold uses SEPARATE __riscv_vfmul/__riscv_vfadd so it is
// contraction-immune; _generic's `sums[l] += d*aux32[l]` fuses to fma under
// -ffp-contract=on/default/fast, so only =off leaves the reference unfused).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

typedef struct {
    union { struct { ggml_half d; ggml_half dmin; }; uint32_t dm; };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

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

// ---- ggml's own reference, VERBATIM (quants.c:645-718). ------------------------
static void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s,
                                           size_t bs, const void * GGML_RESTRICT vx,
                                           size_t bx, const void * GGML_RESTRICT vy,
                                           size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *) vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    std::memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        std::memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ---- The kernel our compiler emitted -------------------------------------------
// void f(size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
extern "C" void
tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

// A NEGATIVE-CONTROL variant of the oracle that perturbs the MIN term: it ADDS
// 1 to dmin's contribution per super-block (sumf -= (dmin+eps)*sumi). It MUST
// make the comparison FAIL whenever the min term is nonzero, proving the harness
// genuinely discriminates the MIN piece (not just the positive fold).
static void ggml_vec_dot_q4_K_q8_K_min_perturbed(int n, float *s, const void *vx,
                                                 const void *vy) {
    const block_q4_K *x = (const block_q4_K *) vx;
    const block_q8_K *y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];
    int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    std::memset(sums, 0, sizeof(sums));
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs; const int8_t *q8 = y[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        int8_t *a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux; utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8; int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int q = 0; q < 4; ++q) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        // PERTURBED: dmin -> dmin + 1.0f (changes the min contribution only).
        const float dmin = (GGML_CPU_FP16_TO_FP32(x[i].dmin) + 1.0f) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

static inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline bool bit_equal(float a, float b) { return bits(a) == bits(b); }

static unsigned g_rng = 0x6b5f1c27u;
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

static void fill_random_block(uint8_t *q4k, uint8_t *q8k) {
    uint16_t d = random_fp16_delta();
    std::memcpy(q4k + 0, &d, 2);                       // fp16 d
    uint16_t dmin = random_fp16_delta();
    std::memcpy(q4k + 2, &dmin, 2);                    // fp16 dmin (NEW vs K4a)
    for (int i = 0; i < 12; ++i) q4k[4 + i] = (uint8_t)(next_rand() & 0xFF); // scales[12]
    for (int i = 0; i < 128; ++i) q4k[16 + i] = (uint8_t)(next_rand() & 0xFF); // qs[128]
    float dy = (float)((int)(next_rand() % 2000) - 1000) / 4096.0f; // fp32 activation d
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)(next_rand() & 0xFF); // qs
    // bsums[16] int16, NONZERO (the min term reads these -- K4a left them 0).
    for (int i = 0; i < 16; ++i) {
        int16_t bs = (int16_t)((int)(next_rand() % 4000) - 2000);
        std::memcpy(q8k + 260 + 2 * i, &bs, 2);
    }
}

// Compare *s for one buffer of NB super-blocks; returns 1 on bit-mismatch.
static int check_n(int NB, const char *label, int verbose) {
    size_t n = (size_t)NB * 256;
    uint8_t *vx = (uint8_t *)malloc((size_t)NB * 144);
    uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
    for (int ib = 0; ib < NB; ++ib)
        fill_random_block(vx + (size_t)ib * 144, vy + (size_t)ib * 292);

    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q4_K_q8_K_generic((int)n, &ref, 0, vx, 0, vy, 0, 1);
    tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(n, &got, vx, vy);

    int fail = !bit_equal(ref, got);
    if (fail || verbose)
        printf("  [%s] n=%zu  ref=%.9g (0x%08x)  tcrv=%.9g (0x%08x)  %s\n",
               label, n, ref, bits(ref), got, bits(got),
               fail ? "MISMATCH" : "bit-exact");
    free(vx); free(vy);
    return fail;
}

// Edge-case block builders (uniform fields for a controlled extreme).
static void build_edge(uint8_t *q4k, uint8_t *q8k, uint8_t qsByte,
                       uint8_t scaleByte, int8_t q8v, int16_t bsumv,
                       uint16_t dfp16, uint16_t dminfp16, float dy) {
    std::memcpy(q4k + 0, &dfp16, 2);
    std::memcpy(q4k + 2, &dminfp16, 2);
    for (int i = 0; i < 12; ++i) q4k[4 + i] = scaleByte;
    for (int i = 0; i < 128; ++i) q4k[16 + i] = qsByte;
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)q8v;
    for (int i = 0; i < 16; ++i) std::memcpy(q8k + 260 + 2 * i, &bsumv, 2);
}

static int check_edge(const char *label, uint8_t qsByte, uint8_t scaleByte,
                      int8_t q8v, int16_t bsumv, uint16_t dfp16,
                      uint16_t dminfp16, float dy) {
    uint8_t q4k[144]; uint8_t q8k[292];
    build_edge(q4k, q8k, qsByte, scaleByte, q8v, bsumv, dfp16, dminfp16, dy);
    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q4_K_q8_K_generic(256, &ref, 0, q4k, 0, q8k, 0, 1);
    tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(256, &got, q4k, q8k);
    int fail = !bit_equal(ref, got);
    printf("  [edge %s] ref=%.9g (0x%08x) tcrv=%.9g (0x%08x) %s\n",
           label, ref, bits(ref), got, bits(got),
           fail ? "MISMATCH" : "bit-exact");
    return fail;
}

int main() {
    int failures = 0;
    int checked = 0;

    printf("INC-24 q4_K K4b byte-exact *s vs ggml _generic (oracle = verbatim _generic)\n");

    // ---- The named n set ---------------------------------------------------------
    printf("named n set:\n");
    const int ns[] = {1, 2, 8, 16, 100, 256}; // *256 = {256,512,2048,4096,25600,65536}
    const char *nlabels[] = {"n=256","n=512","n=2048","n=4096","n=25600","n=65536"};
    for (int i = 0; i < 6; ++i) {
        failures += check_n(ns[i], nlabels[i], 1); ++checked;
    }

    // ---- Edge cases on extremes (incl. all-zero + scale/min extremes) ------------
    printf("edge cases:\n");
    // d magnitudes: fp16 1.0=0x3C00, 0.5=0x3800, 2.0=0x4000.
    failures += check_edge("all-zero (q4=0 scale=0 q8=0 bsum=0 d=dmin=1.0 dy=1.0)",
                           0x00, 0x00, 0, 0, 0x3C00, 0x3C00, 1.0f); ++checked;
    failures += check_edge("q4=0xFF scale=0xFF q8=+127 bsum=+2000 d=2.0 dmin=0.5 dy=-3.5",
                           0xFF, 0xFF, 127, 2000, 0x4000, 0x3800, -3.5f); ++checked;
    failures += check_edge("q4=0x00 scale=0x00 q8=-128 bsum=-2000 d=0.5 dmin=2.0 dy=2.25",
                           0x00, 0x00, -128, -2000, 0x3800, 0x4000, 2.25f); ++checked;
    failures += check_edge("scale=0x3F(max6) min=0x3F q8=+1 bsum=+1000 d=1.0 dmin=1.0 dy=1.0",
                           0xFF, 0x3F, 1, 1000, 0x3C00, 0x3C00, 1.0f); ++checked;
    failures += check_edge("dy=0.0 -> *s must be 0 (both d and dmin terms vanish)",
                           0xAB, 0x55, 33, 777, 0x3C00, 0x3C00, 0.0f); ++checked;
    failures += check_edge("min-only: q4=0 (aux32=0 -> positive term 0) bsum=+1500 dmin=2.0",
                           0x00, 0xFF, 64, 1500, 0x3C00, 0x4000, 1.5f); ++checked;

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

    // ---- Negative control: the MIN-perturbed oracle MUST mismatch the kernel -----
    printf("MIN-term negative control (perturbed dmin must MISMATCH):\n");
    int neg_discriminating = 0, neg_total = 0;
    for (int t = 0; t < 200; ++t) {
        int NB = 1 + (int)(next_rand() % 8);
        size_t n = (size_t)NB * 256;
        uint8_t *vx = (uint8_t *)malloc((size_t)NB * 144);
        uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
        for (int ib = 0; ib < NB; ++ib)
            fill_random_block(vx + (size_t)ib * 144, vy + (size_t)ib * 292);
        float bad = 0.0f, got = 0.0f;
        ggml_vec_dot_q4_K_q8_K_min_perturbed((int)n, &bad, vx, vy);
        tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(n, &got, vx, vy);
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

    printf("\nINC-24 q4_K K4b: %d positive cases checked, %d failures\n",
           checked, failures);
    if (failures == 0)
        printf("RESULT: PASS (*s bit-exact vs ggml _generic for all cases; "
               "MIN-term negative control discriminating)\n");
    else
        printf("RESULT: FAIL\n");
    return failures == 0 ? 0 : 1;
}
