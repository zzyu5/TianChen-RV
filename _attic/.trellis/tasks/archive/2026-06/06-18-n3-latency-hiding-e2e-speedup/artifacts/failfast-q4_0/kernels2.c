// kernels2.c -- decisive follow-up probes (per advisor).
//
//   1. kern_ceiling_kN: raw decode+vwmacc with N INDEPENDENT i16 accumulators,
//      to distinguish a latency-limited ceiling (chain stalls) from a real
//      throughput wall. If these drop well below the 1.5x line (<=713 ns @4096,
//      <=1904 ns @11008) the wall moves; if they stay flat the wall is real.
//
//   2. kern_v2_slide: the ACTUAL V2 lever -- eliminate the per-block scalar
//      extract (vmv.x.s, a pipeline sync) AND the stack round-trip. Each block's
//      vwredsum lands in lane 0 of a vint32m1; we vslideup it into lane j to
//      build an 8-lane integer sum vector with NO scalar extract and NO memory
//      bounce, then one vfcvt + vfmacc(scale) + final vfredusum per group.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#define XS 18
#define YS 34

#define DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl)                                     \
    vuint8m1_t tx_##v0 = __riscv_vle8_v_u8m1((xb) + 2, (vl));                        \
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)((yb) + 2), (vl));           \
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)((yb) + 2 + 16), (vl));      \
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(tx_##v0, 0x0F, (vl))), 8, (vl)); \
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(                                             \
        __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(tx_##v0, 0x04, (vl))), 8, (vl));

// ---- Ceiling with independent i16m2 accumulators (RVV scalable types cannot
// live in a C array, so each accumulator is a named variable). ----
#define MAC(acc, xb, yb) do {                                                       \
    DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);                                        \
    acc = __riscv_vwmacc_vv_i16m2(acc, v0, y0, vl);                                  \
    acc = __riscv_vwmacc_vv_i16m2(acc, v1, y1, vl);                                  \
} while (0)

void kern_ceiling_k2(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32; size_t vl = 16;
    vint16m2_t a0 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a1 = __riscv_vmv_v_x_i16m2(0, vl);
    int ib = 0;
    for (; ib + 2 <= nb; ib += 2) {
        MAC(a0, vx + (size_t)(ib + 0) * XS, vy + (size_t)(ib + 0) * YS);
        MAC(a1, vx + (size_t)(ib + 1) * XS, vy + (size_t)(ib + 1) * YS);
    }
    for (; ib < nb; ++ib) MAC(a0, vx + (size_t)ib * XS, vy + (size_t)ib * YS);
    a0 = __riscv_vadd_vv_i16m2(a0, a1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    *s = (float)__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(a0, z, vl));
}

void kern_ceiling_k4(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32; size_t vl = 16;
    vint16m2_t a0 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a1 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a2 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a3 = __riscv_vmv_v_x_i16m2(0, vl);
    int ib = 0;
    for (; ib + 4 <= nb; ib += 4) {
        MAC(a0, vx + (size_t)(ib + 0) * XS, vy + (size_t)(ib + 0) * YS);
        MAC(a1, vx + (size_t)(ib + 1) * XS, vy + (size_t)(ib + 1) * YS);
        MAC(a2, vx + (size_t)(ib + 2) * XS, vy + (size_t)(ib + 2) * YS);
        MAC(a3, vx + (size_t)(ib + 3) * XS, vy + (size_t)(ib + 3) * YS);
    }
    for (; ib < nb; ++ib) MAC(a0, vx + (size_t)ib * XS, vy + (size_t)ib * YS);
    a0 = __riscv_vadd_vv_i16m2(a0, a1, vl);
    a2 = __riscv_vadd_vv_i16m2(a2, a3, vl);
    a0 = __riscv_vadd_vv_i16m2(a0, a2, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    *s = (float)__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(a0, z, vl));
}

void kern_ceiling_k8(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32; size_t vl = 16;
    vint16m2_t a0 = __riscv_vmv_v_x_i16m2(0, vl), a1 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a2 = __riscv_vmv_v_x_i16m2(0, vl), a3 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a4 = __riscv_vmv_v_x_i16m2(0, vl), a5 = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t a6 = __riscv_vmv_v_x_i16m2(0, vl), a7 = __riscv_vmv_v_x_i16m2(0, vl);
    int ib = 0;
    for (; ib + 8 <= nb; ib += 8) {
        MAC(a0, vx + (size_t)(ib + 0) * XS, vy + (size_t)(ib + 0) * YS);
        MAC(a1, vx + (size_t)(ib + 1) * XS, vy + (size_t)(ib + 1) * YS);
        MAC(a2, vx + (size_t)(ib + 2) * XS, vy + (size_t)(ib + 2) * YS);
        MAC(a3, vx + (size_t)(ib + 3) * XS, vy + (size_t)(ib + 3) * YS);
        MAC(a4, vx + (size_t)(ib + 4) * XS, vy + (size_t)(ib + 4) * YS);
        MAC(a5, vx + (size_t)(ib + 5) * XS, vy + (size_t)(ib + 5) * YS);
        MAC(a6, vx + (size_t)(ib + 6) * XS, vy + (size_t)(ib + 6) * YS);
        MAC(a7, vx + (size_t)(ib + 7) * XS, vy + (size_t)(ib + 7) * YS);
    }
    for (; ib < nb; ++ib) MAC(a0, vx + (size_t)ib * XS, vy + (size_t)ib * YS);
    a0 = __riscv_vadd_vv_i16m2(a0, a1, vl); a2 = __riscv_vadd_vv_i16m2(a2, a3, vl);
    a4 = __riscv_vadd_vv_i16m2(a4, a5, vl); a6 = __riscv_vadd_vv_i16m2(a6, a7, vl);
    a0 = __riscv_vadd_vv_i16m2(a0, a2, vl); a4 = __riscv_vadd_vv_i16m2(a4, a6, vl);
    a0 = __riscv_vadd_vv_i16m2(a0, a4, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    *s = (float)__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(a0, z, vl));
}

// ---- True V2: per-block vredsum gathered via vslideup, no scalar extract ----
// Group of P=8 blocks. Each block's vwredsum result is a vint32m1 with the sum
// in lane 0. We slide it up into lane j of an 8-lane vint32m1 (VLEN=128 => m1
// holds 4 i32; 8 lanes need m2). Use m2 for the gather vector.
void kern_v2_slide(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;
    size_t vl = 16;
    size_t fvl = 8;
    vfloat32m2_t facc = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    int ib = 0;
    for (; ib + P <= nb; ib += P) {
        // Build an 8-lane i32 vector of per-block sumi WITHOUT scalar extract:
        // place block j's vredsum (lane0 of m1) into lane j of an m2 vector.
        vint32m2_t gathered = __riscv_vmv_v_x_i32m2(0, fvl);
        float sc_arr[8];
        for (int j = 0; j < P; ++j) {
            const uint8_t *xb = vx + (size_t)(ib + j) * XS;
            const uint8_t *yb = vy + (size_t)(ib + j) * YS;
            DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);
            vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
            p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
            vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t r = __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl);  // lane0 = sumi
            // slide r (m1) into lane j of gathered (m2): widen via reinterpret +
            // slideup by j. We use a temporary m2 holding r in lane0.
            vint32m2_t rm2 = __riscv_vlmul_ext_v_i32m1_i32m2(r);
            vint32m2_t slid = __riscv_vslideup_vx_i32m2(__riscv_vmv_v_x_i32m2(0, fvl), rm2, (size_t)j, fvl);
            gathered = __riscv_vadd_vv_i32m2(gathered, slid, fvl);
            sc_arr[j] = (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
        }
        vfloat32m2_t vf = __riscv_vfcvt_f_x_v_f32m2(gathered, fvl);
        vfloat32m2_t vsc = __riscv_vle32_v_f32m2((const float *)sc_arr, fvl);
        facc = __riscv_vfmacc_vv_f32m2(facc, vf, vsc, fvl);
    }
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(facc, zf, fvl));
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);
        vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
        sumf += sumi * (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
    }
    *s = sumf;
}
