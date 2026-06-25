// kernels4.c -- the LAST probe (per advisor): transposed / block-as-lane.
//
// kern_transposed: COST-ONLY probe of the one kernel family that lives OUTSIDE
// the RED_k4 per-block-reduction floor. Lane = block index (process P=8 blocks
// at a time). For each of the 32 element-positions we strided-load that byte
// across the 8 blocks (vlse8 with stride = block stride), nibble-split (x) /
// direct (y), and vwmacc into a per-lane i32 accumulator. NO per-block cross-
// lane reduction at all -- each lane already holds one block's full sumi after
// 32 element-position vwmaccs. Then the same cheap vector fp fold V4 uses.
//
// Correctness is IRRELEVANT here (cost-only probe, like CEILING). We only ask:
// does the strided-load penalty alone clear the 1.5x budget (<=713.9 ns @4096,
// <=1904 ns @11008)? It deliberately does fewer-than-correct ops in places where
// that would not change the load/issue cost being measured.
//
// q4_0 weight byte b at offset (2 + e) for e in [0,15] packs element e (low
// nibble, -8) and element e+16 (high nibble, -8). q8_0 activation byte at
// (2 + e) for e in [0,31] is element e directly.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#define XS 18
#define YS 34

void kern_transposed(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    const int P = 8;                 // 8 block-lanes
    size_t pl = 8;                   // vector length = P lanes
    vfloat32m2_t facc = __riscv_vfmv_v_f_f32m2(0.0f, pl);
    float sc[8];
    int ib = 0;
    for (; ib + P <= nb; ib += P) {
        const uint8_t *xbase = vx + (size_t)ib * XS;       // block ib..ib+7, lane=block
        const uint8_t *ybase = vy + (size_t)ib * YS;
        // per-lane (per-block) i32 accumulator: 8 lanes -> m2 i32 at VLEN=128
        vint32m2_t iacc = __riscv_vmv_v_x_i32m2(0, pl);
        // 16 weight bytes -> 32 elements (low + high nibble). Strided load of
        // byte e across the 8 blocks (stride XS) puts block in the lane.
        for (int e = 0; e < 16; ++e) {
            vuint8m1_t wb = __riscv_vlse8_v_u8m1(xbase + 2 + e, XS, pl);
            vint8m1_t lo = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(wb, 0x0F, pl)), 8, pl);
            vint8m1_t hi = __riscv_vsub_vx_i8m1(
                __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(wb, 0x04, pl)), 8, pl);
            // matching activation bytes: element e and element e+16, strided over blocks
            vint8m1_t a_lo = __riscv_vlse8_v_i8m1((const int8_t *)(ybase + 2 + e), YS, pl);
            vint8m1_t a_hi = __riscv_vlse8_v_i8m1((const int8_t *)(ybase + 2 + e + 16), YS, pl);
            // widen-multiply-accumulate into i32 per-lane accumulator.
            // (i8*i8 -> i16 then widen-add into i32; use vwmacc to i16 then widen.)
            vint16m2_t p = __riscv_vwmul_vv_i16m2(lo, a_lo, pl);
            p = __riscv_vwmacc_vv_i16m2(p, hi, a_hi, pl);
            // widen i16 lane-wise into i32 and add (sign-extend low half -- the 8
            // active lanes live in the low 8 i16 lanes -> low 8 i32 lanes).
            vint32m2_t pw = __riscv_vwadd_vx_i32m2(
                __riscv_vlmul_trunc_v_i16m2_i16m1(p), 0, pl);
            iacc = __riscv_vadd_vv_i32m2(iacc, pw, pl);
        }
        for (int j = 0; j < P; ++j) {
            const uint8_t *xb = xbase + (size_t)j * XS, *yb = ybase + (size_t)j * YS;
            sc[j] = (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
        }
        vfloat32m2_t vf = __riscv_vfcvt_f_x_v_f32m2(iacc, pl);
        facc = __riscv_vfmacc_vv_f32m2(facc, vf, __riscv_vle32_v_f32m2(sc, pl), pl);
    }
    vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(facc, zf, pl));
    // scalar tail (cost negligible; correctness not required for this probe)
    for (; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        sumf += (float)*(const _Float16 *)(xb) * (float)*(const _Float16 *)(yb);
    }
    *s = sumf;
}
