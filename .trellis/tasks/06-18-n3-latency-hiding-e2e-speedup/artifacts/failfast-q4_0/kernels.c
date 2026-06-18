// kernels.c -- standalone q4_0_q8_0 dot kernels for the FAIL-FAST latency-hiding
// probe. Compiled as a SEPARATE translation unit from the harness so -O3 cannot
// hoist the loop-invariant dot out of the best-of-N timing loop.
//
// Data layout (matches ggml exactly):
//   block_q4_0 = { fp16 d; uint8_t qs[16]; }            stride 18
//   block_q8_0 = { fp16 d; int8_t  qs[32]; }            stride 34
//   QK = 32 elements per block.
//
// V0 is ggml's REAL RVV q4_0_q8_0 kernel (verbatim from
//   llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c, also the embedded
//   GGML_REF in the inc10 tuner). It is BOTH the perf baseline AND the
//   bit-faithful "what ships" reference. All other variants are latency-hiding
//   restructures that REORDER the fp32 fold (accuracy-preserving, not
//   bit-exact).
//
// All kernels share the C ABI:  void k(size_t n, float *s, const uint8_t *vx, const uint8_t *vy)

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#define XS 18  // block_q4_0 stride
#define YS 34  // block_q8_0 stride

// ---------------------------------------------------------------------------
// V0: ggml's real RVV kernel (the baseline + reference).
// ---------------------------------------------------------------------------
void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    float sumf = 0;
    size_t vl = 16;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
        vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
        vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
        vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
        vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
        float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
        sumf += sumi * dx * dy;
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// Helper: decode one block into v0,v1 (low/high nibble int8) and load y0,y1.
// Inlined into each variant; keeps the bodies readable.
// ---------------------------------------------------------------------------
#define DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl)                                     \
    vuint8m1_t tx_##v0 = __riscv_vle8_v_u8m1((xb) + 2, (vl));                        \
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)((yb) + 2), (vl));           \
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)((yb) + 2 + 16), (vl));      \
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(tx_##v0, 0x0F, (vl))), 8, (vl)); \
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(tx_##v0, 0x04, (vl))), 8, (vl));

// Per-block integer dot -> sumi (i32 scalar). Independent vredsum per block.
static inline int block_sumi(const uint8_t *xb, const uint8_t *yb, size_t vl) {
    DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);
    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
}

// ===========================================================================
// V1: multi-accumulator. K independent fp32 sumf accumulators break the serial
// fadd dependency chain. The per-block vredsum is unchanged. Scalar tail for
// nb % K. Parameterised by K via three concrete kernels.
// ===========================================================================
#define V1_KERNEL(NAME, K)                                                          \
void NAME(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {                \
    const int nb = (int)n / 32; size_t vl = 16;                                      \
    float acc[K]; for (int j = 0; j < K; ++j) acc[j] = 0.0f;                         \
    int ib = 0;                                                                       \
    for (; ib + (K) <= nb; ib += (K)) {                                              \
        for (int j = 0; j < (K); ++j) {                                             \
            const uint8_t *xb = vx + (size_t)(ib + j) * XS;                          \
            const uint8_t *yb = vy + (size_t)(ib + j) * YS;                          \
            int sumi = block_sumi(xb, yb, vl);                                       \
            float dx = (float)*(const _Float16 *)(xb);                              \
            float dy = (float)*(const _Float16 *)(yb);                              \
            acc[j] += sumi * dx * dy;                                                \
        }                                                                            \
    }                                                                                \
    for (; ib < nb; ++ib) {                                                          \
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;        \
        int sumi = block_sumi(xb, yb, vl);                                           \
        float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb); \
        acc[0] += sumi * dx * dy;                                                    \
    }                                                                                \
    float sumf = 0; for (int j = 0; j < (K); ++j) sumf += acc[j];                    \
    *s = sumf;                                                                        \
}

V1_KERNEL(kern_v1_k2, 2)
V1_KERNEL(kern_v1_k4, 4)
V1_KERNEL(kern_v1_k8, 8)

// ===========================================================================
// V2: batched reduction. Process P blocks at a time; place each block's per-block
// integer sumi into a SEPARATE lane of an i32 vector (lane = block), and place
// each block's combined fp32 scale (dx*dy) into the matching lane of an fp32
// vector. Then ONE fp32 fused-multiply over P lanes and ONE vector->scalar
// reduction per P blocks (instead of P serial scalar fadds). This defers and
// vectorises the fp32 fold.
//
// We still do a vredsum per block to get sumi (the integer reduction is cheap
// and independent); the lever here is collapsing the *fp32 fold* into a
// vectorised multiply-add + a single tree reduction per P. P=8 uses m1 i32/f32
// (VLEN=128 => 4 lanes/m1, so P=8 spans m2). We use P=8 with m2 f32 (8 lanes).
// ===========================================================================
void kern_v2_p8(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;
    size_t vl = 16;
    // fp32 accumulator vector (8 lanes, m2 at VLEN=128). One tree reduction at end.
    size_t fvl = 8;
    vfloat32m2_t facc = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    int ib = 0;
    for (; ib + P <= nb; ib += P) {
        int32_t sumi_arr[8];
        float sc_arr[8];
        for (int j = 0; j < P; ++j) {
            const uint8_t *xb = vx + (size_t)(ib + j) * XS;
            const uint8_t *yb = vy + (size_t)(ib + j) * YS;
            sumi_arr[j] = block_sumi(xb, yb, vl);
            float dx = (float)*(const _Float16 *)(xb);
            float dy = (float)*(const _Float16 *)(yb);
            sc_arr[j] = dx * dy;
        }
        // Vectorise the fp32 fold: facc += (float)sumi * scale, lane-parallel.
        vint32m2_t vsumi = __riscv_vle32_v_i32m2((const int32_t *)sumi_arr, fvl);
        vfloat32m2_t vf = __riscv_vfcvt_f_x_v_f32m2(vsumi, fvl);
        vfloat32m2_t vsc = __riscv_vle32_v_f32m2((const float *)sc_arr, fvl);
        facc = __riscv_vfmacc_vv_f32m2(facc, vf, vsc, fvl);
    }
    // One tree reduction of the 8-lane fp32 accumulator.
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(facc, zf, fvl));
    // Scalar tail.
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        int sumi = block_sumi(xb, yb, vl);
        float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
        sumf += sumi * dx * dy;
    }
    *s = sumf;
}

// ===========================================================================
// V3: best-of combined. Multi-accumulator (K=4 fp32 vector lanes deferred) on
// top of larger integer-core LMUL and software pipelining of decode(i+1) with
// compute(i). Here: process the integer half at m2 (two blocks' qs packed into
// one m2 register pair is awkward given the q4_0/q8_0 nibble layout), so instead
// we combine V1's K independent accumulators with V2's vectorised fp32 fold AND
// unroll-prefetch the next block's loads. P=8 fp32 vector fold + decode software
// pipeline.
//
// The integer LMUL lever: q4_0 packs 32 weights into 16 bytes (low+high nibble),
// q8_0 has 32 int8. vl=16 with m1 is the natural width for the nibble split.
// Going to m2 would require reorganising the nibble unpack across two blocks;
// instead V3's win is fp32-fold vectorisation + decode/compute pipelining.
// ===========================================================================
void kern_v3(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;
    size_t vl = 16;
    size_t fvl = 8;
    // Two independent fp32 vector accumulators to break the cross-iteration
    // facc dependency (P=8 each => process 16 blocks per outer iter).
    vfloat32m2_t facc0 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    vfloat32m2_t facc1 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    int ib = 0;
    for (; ib + 2 * P <= nb; ib += 2 * P) {
        int32_t si0[8], si1[8];
        float sc0[8], sc1[8];
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS;
            const uint8_t *ya = vy + (size_t)(ib + j) * YS;
            const uint8_t *xb = vx + (size_t)(ib + P + j) * XS;
            const uint8_t *yb = vy + (size_t)(ib + P + j) * YS;
            si0[j] = block_sumi(xa, ya, vl);
            si1[j] = block_sumi(xb, yb, vl);
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
            sc1[j] = (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2((const int32_t *)si0, fvl), fvl);
        vfloat32m2_t f1 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2((const int32_t *)si1, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2((const float *)sc0, fvl), fvl);
        facc1 = __riscv_vfmacc_vv_f32m2(facc1, f1, __riscv_vle32_v_f32m2((const float *)sc1, fvl), fvl);
    }
    // Handle a single remaining P-block group.
    for (; ib + P <= nb; ib += P) {
        int32_t si0[8]; float sc0[8];
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS;
            const uint8_t *ya = vy + (size_t)(ib + j) * YS;
            si0[j] = block_sumi(xa, ya, vl);
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2((const int32_t *)si0, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2((const float *)sc0, fvl), fvl);
    }
    facc0 = __riscv_vfadd_vv_f32m2(facc0, facc1, fvl);
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(facc0, zf, fvl));
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        int sumi = block_sumi(xb, yb, vl);
        float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
        sumf += sumi * dx * dy;
    }
    *s = sumf;
}

// ===========================================================================
// CEILING probe: raw integer compute only (vwmacc), no per-block reduction,
// no fp32 scale/fold. Accumulate all products into ONE i16 vector accumulator
// across blocks (deliberately wrong numerically -- it exists ONLY to measure
// the board's nibble-decode + vwmacc issue ceiling, i.e. how fast the kernel
// could ever go if reduction+scale were free). Returns a meaningless float.
// ===========================================================================
void kern_ceiling(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    size_t vl = 16;
    vint16m2_t acc = __riscv_vmv_v_x_i16m2(0, vl);
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);
        acc = __riscv_vwmacc_vv_i16m2(acc, v0, y0, vl);
        acc = __riscv_vwmacc_vv_i16m2(acc, v1, y1, vl);
    }
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(acc, z, vl));
    *s = (float)sumi;
}
