// kernels3.c -- discriminating probe + candidate real kernel (per advisor).
//
//   A. kern_red_k4: ceiling_k4 STRUCTURE but with a per-block vwredsum whose
//      result is accumulated into K=4 INDEPENDENT vint32m1 accumulators via
//      vadd. NO vmv.x.s, NO scale, NO fp. Isolates: does the per-block vredsum
//      pipeline (stays ~>=1.7x) or is it the serial wall (collapses ~1.3x)?
//
//   B. kern_v4: the candidate accuracy-preserving real kernel. Process P=8
//      blocks per group: 8 INDEPENDENT vwredsums (each lands sumi in lane0 of a
//      vint32m1), then vse32 lane0 of each to a contiguous 8-int32 buffer
//      (vector stores, NO GPR sync / NO vmv.x.s), one vle32 -> vfcvt -> vfmacc
//      against the 8 per-block scales, into K=2 independent fp32 vector
//      accumulators. One vfredusum at the very end. This is V2_p8 minus the
//      scalar extract and minus the serial slide -- the two things that sank it.

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

// Per-block: decode + vwmul/vwmacc + vwredsum -> vint32m1 (sumi in lane0).
#define BLOCK_REDSUM(xb, yb, vl, out)                                               \
    vint32m1_t out;                                                                 \
    do {                                                                            \
        DECODE_BLOCK(xb, yb, v0, v1, y0, y1, vl);                                    \
        vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);                           \
        p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);                                  \
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);                                 \
        out = __riscv_vwredsum_vs_i16m2_i32m1(p, z, vl);                             \
    } while (0)

// ---- A. Discriminator: per-block vredsum into K=4 independent i32 accumulators
//          (no scalar extract, no scale, no fp). ----
void kern_red_k4(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32; size_t vl = 16;
    vint32m1_t r0 = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r1 = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r2 = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r3 = __riscv_vmv_v_x_i32m1(0, vl);
    int ib = 0;
    for (; ib + 4 <= nb; ib += 4) {
        { BLOCK_REDSUM(vx + (size_t)(ib+0)*XS, vy + (size_t)(ib+0)*YS, vl, b0); r0 = __riscv_vadd_vv_i32m1(r0, b0, vl); }
        { BLOCK_REDSUM(vx + (size_t)(ib+1)*XS, vy + (size_t)(ib+1)*YS, vl, b1); r1 = __riscv_vadd_vv_i32m1(r1, b1, vl); }
        { BLOCK_REDSUM(vx + (size_t)(ib+2)*XS, vy + (size_t)(ib+2)*YS, vl, b2); r2 = __riscv_vadd_vv_i32m1(r2, b2, vl); }
        { BLOCK_REDSUM(vx + (size_t)(ib+3)*XS, vy + (size_t)(ib+3)*YS, vl, b3); r3 = __riscv_vadd_vv_i32m1(r3, b3, vl); }
    }
    for (; ib < nb; ++ib) { BLOCK_REDSUM(vx + (size_t)ib*XS, vy + (size_t)ib*YS, vl, b); r0 = __riscv_vadd_vv_i32m1(r0, b, vl); }
    r0 = __riscv_vadd_vv_i32m1(r0, r1, vl);
    r2 = __riscv_vadd_vv_i32m1(r2, r3, vl);
    r0 = __riscv_vadd_vv_i32m1(r0, r2, vl);
    *s = (float)__riscv_vmv_x_s_i32m1_i32(r0);
}

// ---- B. Candidate real kernel V4: 8 independent vredsums -> contiguous vector
//          store (no scalar extract) -> vle32 -> vfcvt -> vfmacc with scales,
//          2 independent fp32 accumulators, one vfredusum at end. ----
void kern_v4(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;
    size_t vl = 16;
    size_t fvl = 8;
    vfloat32m2_t facc0 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    vfloat32m2_t facc1 = __riscv_vfmv_v_f_f32m2(0.0f, fvl);
    int32_t buf0[8], buf1[8];
    float sc0[8], sc1[8];
    int ib = 0;
    // Process 2*P=16 blocks per outer iter -> 2 independent fp32 accumulators.
    for (; ib + 2 * P <= nb; ib += 2 * P) {
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS, *ya = vy + (size_t)(ib + j) * YS;
            const uint8_t *xb = vx + (size_t)(ib + P + j) * XS, *yb = vy + (size_t)(ib + P + j) * YS;
            BLOCK_REDSUM(xa, ya, vl, ra);
            BLOCK_REDSUM(xb, yb, vl, rb);
            // store lane0 of each redsum to contiguous buffer (vse32 vl=1; no GPR sync)
            __riscv_vse32_v_i32m1(buf0 + j, ra, 1);
            __riscv_vse32_v_i32m1(buf1 + j, rb, 1);
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
            sc1[j] = (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf0, fvl), fvl);
        vfloat32m2_t f1 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf1, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2(sc0, fvl), fvl);
        facc1 = __riscv_vfmacc_vv_f32m2(facc1, f1, __riscv_vle32_v_f32m2(sc1, fvl), fvl);
    }
    for (; ib + P <= nb; ib += P) {
        for (int j = 0; j < P; ++j) {
            const uint8_t *xa = vx + (size_t)(ib + j) * XS, *ya = vy + (size_t)(ib + j) * YS;
            BLOCK_REDSUM(xa, ya, vl, ra);
            __riscv_vse32_v_i32m1(buf0 + j, ra, 1);
            sc0[j] = (float)*(const _Float16 *)(xa) * (float)*(const _Float16 *)(ya);
        }
        vfloat32m2_t f0 = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(buf0, fvl), fvl);
        facc0 = __riscv_vfmacc_vv_f32m2(facc0, f0, __riscv_vle32_v_f32m2(sc0, fvl), fvl);
    }
    facc0 = __riscv_vfadd_vv_f32m2(facc0, facc1, fvl);
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(facc0, zf, fvl));
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        BLOCK_REDSUM(xb, yb, vl, r);
        int sumi = __riscv_vmv_x_s_i32m1_i32(r);
        sumf += sumi * (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
    }
    *s = sumf;
}
