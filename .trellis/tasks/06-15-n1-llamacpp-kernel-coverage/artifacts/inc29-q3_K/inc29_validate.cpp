// INC-29 (q3_K) byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new typed op
// tcrv_rvv.q3_k_q8_k_block_dot computes the SAME fp32 dot-product output *s as
// ggml's own reference ggml_vec_dot_q3_K_q8_K_generic (llama.cpp/ggml/src/
// ggml-cpu/quants.c:566-643), BYTE-EXACT (exact IEEE-754 bit equality on the
// fp32 *s), over the full n set + edge cases incl. (a) a PER-PLANE hmask test
// that independently toggles all 8 high-bit planes p=0..7, (b) a SUBTRACTIVE-
// hmask negative control (inverts the hmask polarity), and (c) a SIGNED-scale
// negative control (drops the -32 bias).
//
// ORACLE = the VERBATIM ggml _generic function body (copied below, NOT a
// re-derivation). The one external macro GGML_CPU_FP16_TO_FP32 is shimmed to a
// hardware _Float16 cast (exact for every fp16: fp16->fp32 widening never
// rounds). q3_K = the 3-bit modern K-quant, the LAST common K-quant: 16
// sub-blocks of 16, 2-bit weights from qs (4 per byte) + a SUBTRACTIVE high bit
// from the 32-byte hmask plane (signed [-4,3]), 6-bit SIGNED scales (scales[12]
// dance, scales[j]-32), and q6_K's NO-min deferred d.Sum(aux32) fold.
//
// Block byte layouts (ggml-common.h, QK_K = 256):
//   block_q3_K (110 bytes): hmask[32]@0 | qs[64]@32 | scales[12]@96 | d(fp16)@108
//   block_q8_K (292 bytes): d(fp32)@0 | qs[256]@4 | bsums[16]@260 (bsums UNUSED)
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.
// Headline byte-exact regime: BOTH oracle and kernel compiled -ffp-contract=off.
// The fold is q6_K's deferred lane-wise `sums[l] += d*aux32[l]` (SEPARATE
// mul/add, not fma) + a SEQUENTIAL horizontal sum, exactly matching _generic's
// `sums[l] += d*aux32[l]` + `for(l) sumf += sums[l]`, so it is byte-exact under
// -ffp-contract=off. Reported across off/on/fast (a contraction band may appear
// under on/fast since aux32[l]->float feeds a mul then an add; noted if so).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#define QK_K 256

typedef uint16_t ggml_half;

typedef struct {
    uint8_t hmask[QK_K/8]; // 32 bytes, the high-bit plane (256 bits)
    uint8_t qs[QK_K/4];    // 64 bytes, the low 2 bits (4 per byte)
    uint8_t scales[12];    // 12 packed 6-bit-signed-scale bytes
    ggml_half d;           // fp16 super-block scale
} block_q3_K;

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

// ---- ggml's own reference, VERBATIM (quants.c:566-643). ------------------------
static void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s,
                                           size_t bs, const void * GGML_RESTRICT vx,
                                           size_t bx, const void * GGML_RESTRICT vy,
                                           size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = (const block_q3_K *) vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// ---- The kernel our compiler emitted -------------------------------------------
// void f(size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
extern "C" void
tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

// A NEGATIVE-CONTROL variant that INVERTS the hmask polarity: `a -= (hm&m ? 4 :
// 0)` instead of `a -= (hm&m ? 0 : 4)`. It MUST mismatch the kernel whenever the
// hmask plane is non-uniform -- proving the SUBTRACTIVE direction (the q3_K
// distinguishing high-bit decode) is load-bearing.
static void ggml_vec_dot_q3_K_q8_K_hmask_inverted(int n, float *s, const void *vx,
                                                  const void *vy) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q3_K *x = (const block_q3_K *) vx;
    const block_q8_K *y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    uint32_t auxs[4]; const int8_t *scales = (const int8_t*)auxs;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = x[i].qs; const uint8_t *hm = x[i].hmask;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t *a = aux8; uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 4 : 0); // INVERTED
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 4 : 0); // INVERTED
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 4 : 0); // INVERTED
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 4 : 0); // INVERTED
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// A NEGATIVE-CONTROL variant that DROPS the signed -32 scale bias (uses scales[j]
// instead of scales[j]-32). It MUST mismatch the kernel whenever any aux32 lane
// is nonzero -- proving the SIGNED 6-bit scale (the -32 bias from the q3_K dance)
// is load-bearing.
static void ggml_vec_dot_q3_K_q8_K_no_scale_bias(int n, float *s, const void *vx,
                                                 const void *vy) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q3_K *x = (const block_q3_K *) vx;
    const block_q8_K *y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    uint32_t auxs[4]; const int8_t *scales = (const int8_t*)auxs;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = x[i].qs; const uint8_t *hm = x[i].hmask;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t *a = aux8; uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j]) * aux16[l]; // NO -32
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j]) * aux16[l]; // NO -32
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

static inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline bool bit_equal(float a, float b) { return bits(a) == bits(b); }

static unsigned g_rng = 0x3d7a91c5u;
static unsigned next_rand() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

static uint16_t random_fp16_delta() {
    uint16_t mant = (uint16_t)(next_rand() & 0x3FF);
    uint16_t exp  = (uint16_t)(8 + (next_rand() % 12)); // ~2^-7 .. 2^4
    return (uint16_t)((exp << 10) | mant);
}

static void fill_random_block(uint8_t *q3k, uint8_t *q8k) {
    for (int i = 0; i < 32; ++i) q3k[0 + i]  = (uint8_t)(next_rand() & 0xFF); // hmask[32]
    for (int i = 0; i < 64; ++i) q3k[32 + i] = (uint8_t)(next_rand() & 0xFF); // qs[64]
    for (int i = 0; i < 12; ++i) q3k[96 + i] = (uint8_t)(next_rand() & 0xFF); // scales[12]
    uint16_t d = random_fp16_delta();
    std::memcpy(q3k + 108, &d, 2);                      // fp16 d
    float dy = (float)((int)(next_rand() % 2000) - 1000) / 4096.0f; // fp32 act d
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)(next_rand() & 0xFF); // qs
    for (int i = 0; i < 16; ++i) {
        int16_t bs = (int16_t)((int)(next_rand() % 4000) - 2000);
        std::memcpy(q8k + 260 + 2 * i, &bs, 2);          // bsums[16] (unused by q3_K)
    }
}

// Compare *s for one buffer of NB super-blocks; returns 1 on bit-mismatch.
static int check_n(int NB, const char *label, int verbose) {
    size_t n = (size_t)NB * 256;
    uint8_t *vx = (uint8_t *)malloc((size_t)NB * 110);
    uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
    for (int ib = 0; ib < NB; ++ib)
        fill_random_block(vx + (size_t)ib * 110, vy + (size_t)ib * 292);

    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q3_K_q8_K_generic((int)n, &ref, 0, vx, 0, vy, 0, 1);
    tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(n, &got, vx, vy);

    int fail = !bit_equal(ref, got);
    if (fail || verbose)
        printf("  [%s] n=%zu  ref=%.9g (0x%08x)  tcrv=%.9g (0x%08x)  %s\n",
               label, n, ref, bits(ref), got, bits(got),
               fail ? "MISMATCH" : "bit-exact");
    free(vx); free(vy);
    return fail;
}

// Edge-case block builder (uniform-ish fields for a controlled extreme; hmask and
// scales are explicit byte patterns).
static void build_edge(uint8_t *q3k, uint8_t *q8k, uint8_t qsByte,
                       uint8_t hmaskByte, const uint8_t scales12[12],
                       int8_t q8v, uint16_t dfp16, float dy) {
    for (int i = 0; i < 32; ++i)  q3k[0 + i]  = hmaskByte;
    for (int i = 0; i < 64; ++i)  q3k[32 + i] = qsByte;
    for (int i = 0; i < 12; ++i)  q3k[96 + i] = scales12[i];
    std::memcpy(q3k + 108, &dfp16, 2);
    std::memcpy(q8k + 0, &dy, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)q8v;
    for (int i = 0; i < 16; ++i) { int16_t z = 0; std::memcpy(q8k + 260 + 2*i, &z, 2); }
}

static int check_edge(const char *label, uint8_t qsByte, uint8_t hmaskByte,
                      const uint8_t scales12[12], int8_t q8v, uint16_t dfp16,
                      float dy) {
    uint8_t q3k[110]; uint8_t q8k[292];
    build_edge(q3k, q8k, qsByte, hmaskByte, scales12, q8v, dfp16, dy);
    float ref = 0.0f, got = 0.0f;
    ggml_vec_dot_q3_K_q8_K_generic(256, &ref, 0, q3k, 0, q8k, 0, 1);
    tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(256, &got, q3k, q8k);
    int fail = !bit_equal(ref, got);
    printf("  [edge %s] ref=%.9g (0x%08x) tcrv=%.9g (0x%08x) %s\n",
           label, ref, bits(ref), got, bits(got),
           fail ? "MISMATCH" : "bit-exact");
    return fail;
}

// The PER-PLANE hmask discriminator: for each bit-plane p in 0..7, set ONLY plane
// p in the hmask of element 0 (i.e. hmask[0] = 1<<p, all else 0). Sub-block 0
// covers aux8[0..15] which is the first element-group of plane p's shift only for
// the SHIFT that p selects; setting q8 nonzero makes the plane-p high bit on
// element 0 visible in *s. This independently exercises each of the 8 planes so a
// mis-mapping of p>=4 (chunk 1) cannot pass. We compare the kernel to _generic on
// per-plane data (a POSITIVE check), AND verify the hmask-inverted oracle DIFFERS
// from the kernel on the same data (a discrimination check).
static int check_per_plane_hmask() {
    int fail = 0;
    printf("per-plane hmask (each of 8 bit-planes toggled independently):\n");
    for (int p = 0; p < 8; ++p) {
        uint8_t q3k[110] = {0}; uint8_t q8k[292] = {0};
        // hmask: set plane p for EVERY element (hm[l] = 1<<p for all 32 l) so the
        // plane-p high bit fires on every element whose shift maps to plane p.
        for (int l = 0; l < 32; ++l) q3k[0 + l] = (uint8_t)(1u << p);
        // qs: a fixed pattern with all four 2-bit fields = 1 (0x55) so the low
        // bits are nonzero and the high bit lifts each affected element by +4.
        for (int i = 0; i < 64; ++i) q3k[32 + i] = 0x55;
        // scales: all 0x21 nibbles -> 6-bit values 0x21=33 -> scale 33-32 = +1
        // (every sub-block contributes with scale +1; signed bias exercised).
        for (int i = 0; i < 12; ++i) q3k[96 + i] = 0x21;
        uint16_t d = 0x3C00; std::memcpy(q3k + 108, &d, 2);   // fp16 1.0
        float dy = 1.0f; std::memcpy(q8k + 0, &dy, 4);
        for (int i = 0; i < 256; ++i) q8k[4 + i] = 7;          // q8 = +7
        float ref = 0.0f, got = 0.0f, inv = 0.0f;
        ggml_vec_dot_q3_K_q8_K_generic(256, &ref, 0, q3k, 0, q8k, 0, 1);
        tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(256, &got, q3k, q8k);
        ggml_vec_dot_q3_K_q8_K_hmask_inverted(256, &inv, q3k, q8k);
        int pf = !bit_equal(ref, got);
        int discr = !bit_equal(inv, got); // inverted MUST differ (plane p is set)
        printf("  [plane p=%d] ref=%.9g tcrv=%.9g %s | inverted=%.9g %s\n",
               p, ref, got, pf ? "MISMATCH" : "bit-exact",
               inv, discr ? "(discriminates)" : "(NO-DISCRIMINATE!)");
        fail += pf;
        if (!discr) { printf("  PER-PLANE DISCRIMINATION FAILED at p=%d!\n", p); ++fail; }
    }
    return fail;
}

int main() {
    int failures = 0;
    int checked = 0;

    printf("INC-29 q3_K byte-exact *s vs ggml _generic (oracle = verbatim _generic)\n");

    // ---- The named n set ---------------------------------------------------------
    printf("named n set:\n");
    const int ns[] = {1, 2, 8, 16, 100, 256}; // *256 = {256,512,2048,4096,25600,65536}
    const char *nlabels[] = {"n=256","n=512","n=2048","n=4096","n=25600","n=65536"};
    for (int i = 0; i < 6; ++i) {
        failures += check_n(ns[i], nlabels[i], 1); ++checked;
    }

    // ---- Edge cases on extremes --------------------------------------------------
    printf("edge cases:\n");
    uint8_t sc_zero[12]; for (int i = 0; i < 12; ++i) sc_zero[i] = 0x00;
    uint8_t sc_ff[12];   for (int i = 0; i < 12; ++i) sc_ff[i]   = 0xFF;
    uint8_t sc_20[12];   for (int i = 0; i < 12; ++i) sc_20[i]   = 0x20; // 6-bit 32 -> scale 0
    // all-zero hmask: every high bit UNSET -> a = low2 - 4 in [-4,-1] (the most
    // negative q3 lane). qs=0 -> low2=0 -> a = -4 everywhere. scales=0x20 -> scale
    // 32-32 = 0 -> *s = 0 (scale annihilates). Tests scale=0 + a=-4.
    failures += check_edge("hmask=0 (all UNSET -> a=low2-4) qs=0 scale=0 (sc=0x20) q8=+1 d=1.0 dy=1.0",
                           0x00, 0x00, sc_20, 1, 0x3C00, 1.0f); ++checked;
    // all-FF hmask: every high bit SET -> a = low2 (in [0,3]); qs=0xFF -> low2=3 ->
    // a = 3 everywhere (the most positive q3 lane). scales=0xFF: 6-bit dance.
    failures += check_edge("hmask=0xFF (all SET -> a=low2=3) qs=0xFF scale=0xFF q8=+127 d=2.0 dy=-3.5",
                           0xFF, 0xFF, sc_ff, 127, 0x4000, -3.5f); ++checked;
    // q3 = +3 EVERYWHERE (hmask all SET, qs=0xFF): the +3 extreme of [-4,3].
    failures += check_edge("q3=+3 (hmask=0xFF, qs=0xFF) scale=0x00 (sc-32=-32) q8=-128 d=0.5 dy=2.25",
                           0xFF, 0xFF, sc_zero, -128, 0x3800, 2.25f); ++checked;
    // q3 = -4 EVERYWHERE (hmask all UNSET, qs=0): the -4 extreme of [-4,3].
    failures += check_edge("q3=-4 (hmask=0, qs=0) scale=0xFF q8=+100 d=1.5 dy=1.0",
                           0x00, 0x00, sc_ff, 100, 0x3E00, 1.0f); ++checked;
    // mixed qs=0x1B (fields 3,2,1,0 by shift), hmask=0xA5 (planes 0,2,5,7 set).
    failures += check_edge("qs=0x1B (fields by shift) hmask=0xA5 (planes 0,2,5,7) scale=0x55 q8=+3 d=0.5 dy=1.0",
                           0x1B, 0xA5, sc_ff, 3, 0x3800, 1.0f); ++checked;
    // dy = 0 -> *s must be exactly 0 (d vanishes).
    failures += check_edge("dy=0.0 -> *s must be 0 (d vanishes)",
                           0xAB, 0x5A, sc_20, 33, 0x3C00, 0.0f); ++checked;

    // ---- Per-plane hmask discriminator (all 8 planes p=0..7) ---------------------
    {
        int pf = check_per_plane_hmask();
        failures += pf; checked += 8;
    }

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

    // ---- Negative control 1: hmask-INVERTED oracle MUST mismatch the kernel ------
    printf("SUBTRACTIVE-hmask negative control (inverted polarity must MISMATCH):\n");
    int hm_discriminating = 0, hm_total = 0;
    for (int t = 0; t < 200; ++t) {
        int NB = 1 + (int)(next_rand() % 8);
        size_t n = (size_t)NB * 256;
        uint8_t *vx = (uint8_t *)malloc((size_t)NB * 110);
        uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
        for (int ib = 0; ib < NB; ++ib) {
            uint8_t *bx = vx + (size_t)ib * 110;
            fill_random_block(bx, vy + (size_t)ib * 292);
            // Force a non-uniform hmask so the polarity inversion diverges
            // (at least one bit set AND one unset across the planes).
            for (int i = 0; i < 32; ++i) bx[0 + i] = (uint8_t)(next_rand() & 0xFF);
        }
        float bad = 0.0f, got = 0.0f;
        ggml_vec_dot_q3_K_q8_K_hmask_inverted((int)n, &bad, vx, vy);
        tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(n, &got, vx, vy);
        if (!bit_equal(bad, got)) ++hm_discriminating;
        ++hm_total;
        free(vx); free(vy);
    }
    printf("  hmask-inverted-vs-kernel mismatched: %d / %d (must be %d)\n",
           hm_discriminating, hm_total, hm_total);
    if (hm_discriminating != hm_total) {
        printf("  SUBTRACTIVE-hmask NEGATIVE CONTROL FAILED: kernel ignores hmask polarity!\n");
        ++failures;
    }

    // ---- Negative control 2: signed-scale-DROPPING oracle MUST mismatch ----------
    printf("SIGNED-scale negative control (no -32 bias must MISMATCH):\n");
    int sc_discriminating = 0, sc_total = 0;
    for (int t = 0; t < 200; ++t) {
        int NB = 1 + (int)(next_rand() % 8);
        size_t n = (size_t)NB * 256;
        uint8_t *vx = (uint8_t *)malloc((size_t)NB * 110);
        uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
        for (int ib = 0; ib < NB; ++ib)
            fill_random_block(vx + (size_t)ib * 110, vy + (size_t)ib * 292);
        float bad = 0.0f, got = 0.0f;
        ggml_vec_dot_q3_K_q8_K_no_scale_bias((int)n, &bad, vx, vy);
        tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(n, &got, vx, vy);
        if (!bit_equal(bad, got)) ++sc_discriminating;
        ++sc_total;
        free(vx); free(vy);
    }
    printf("  no-scale-bias-vs-kernel mismatched: %d / %d (must be %d)\n",
           sc_discriminating, sc_total, sc_total);
    if (sc_discriminating != sc_total) {
        printf("  SIGNED-scale NEGATIVE CONTROL FAILED: kernel ignores the -32 bias!\n");
        ++failures;
    }

    printf("\nINC-29 q3_K: %d positive cases checked, %d failures\n",
           checked, failures);
    if (failures == 0)
        printf("RESULT: PASS (*s bit-exact vs ggml _generic for all cases; "
               "per-plane hmask + SUBTRACTIVE-hmask + SIGNED-scale controls discriminating)\n");
    else
        printf("RESULT: FAIL\n");
    return failures == 0 ? 0 : 1;
}
