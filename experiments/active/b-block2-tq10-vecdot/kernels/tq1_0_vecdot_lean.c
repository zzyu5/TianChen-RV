/* tq1_0_vecdot_lean.c -- OWNED LEAN ternary vec_dot leaf: tq1_0 x q8_K (M=1).
 *
 * B线第二块 P1. v2: adopts the ggml hand-tuned _vl128 opponent's READABLE leaner
 * techniques (the objdump-recon target structure) on top of the v1 fused leaf:
 *   (1) digit-0 skips the vmul_vx(*1)  (pow3[0]==1 is a no-op);
 *   (2) the trit is kept in i16 directly via vsub_vx_u16(tq,1)+reinterpret -- NO
 *       narrow-to-u8 / vadd_i8 round-trip;
 *   (3) q8 is pre-widened to i16 (vwcvt) once per load, and the plane products
 *       accumulate with vmul_vv (digit-0 init) + vmacc_vv (digits 1..4) into ONE
 *       i16 accumulator (no aux8, matches opponent's mac chain);
 *   (4) the qh region is decoded in a SINGLE vectorized pass via a pow[16] lookup
 *       (broadcast the 4 qh bytes across the register, one vmul_vv), instead of the
 *       v1 four separate qh passes.
 * ONE final vwredsum per super-block (i16 accumulators folded then reduced once).
 *
 * PURE C-INTRINSIC. NO inline-asm, NO pinned schedule. This is an OWNED authored
 * re-expression of the opponent's readable structure -- parity here = path-win by
 * reproducing the optimal readable form (NOT a byte copy of ggml's source).
 *
 * BYTE-EXACT integer core to ggml_vec_dot_tq1_0_q8_K_generic: identical pow3 mod-256
 * decode, identical q8 index map, order-free integer sum. i16 products never overflow
 * (|trit|<=1, |q8|<=127 -> |product|<=127; per-lane <=5 plane products <= 635).
 */
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

#ifdef __cplusplus
extern "C"
#endif
void weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const size_t nb = n / 256;
  static const uint8_t pow16[16] = {1,1,1,1, 3,3,3,3, 9,9,9,9, 27,27,27,27};
  float sumf = 0.0f;

  for (size_t ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + ib * 54;
    const uint8_t *yb = vy + ib * 292;
    const uint8_t *tq = xb;                 /* qs[0..47], qh[48..51] */
    const int8_t  *q8 = (const int8_t *)(yb + 4);

    /* ---- (A) main qs region: 32 lanes, i16m4, digits 0..4 -> q8[0..159] ---- */
    vint16m4_t suml1;
    {
      const size_t vl = 32;
      vuint8m2_t tqb = __riscv_vle8_v_u8m2(tq, vl);
      /* digit 0 (pow3=1, no vmul_vx): xi = (tqb*3)>>8; trit = xi-1 (u16 wrap+reinterp) */
      vuint16m4_t tq0 = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(tqb, 3, vl), 8, vl);
      vint16m4_t  tr0 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tq0, 1, vl));
      vint16m4_t  q80 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
      suml1 = __riscv_vmul_vv_i16m4(tr0, q80, vl);
      uint8_t p = 3;
      for (int t = 0; t < 4; ++t) {         /* digits 1..4 (pow3 = 3,9,27,81) */
        vuint16m4_t tqn = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vx_u8m2(tqb, p, vl), 3, vl), 8, vl);
        vint16m4_t  trn = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tqn, 1, vl));
        vint16m4_t  q8n = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8 + (t + 1) * 32, vl), vl);
        suml1 = __riscv_vmacc_vv_i16m4(suml1, trn, q8n, vl);
        p *= 3;
      }
    }

    /* ---- (B) tail qs region: 16 lanes, i16m2, digits 0..4 -> q8[160..239] ---- */
    vint16m2_t suml2;
    {
      const size_t vl = 16;
      vuint8m1_t tqb = __riscv_vle8_v_u8m1(tq + 32, vl);
      const int8_t *q8t = q8 + 160;
      vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tqb, 3, vl), 8, vl);
      vint16m2_t  tr0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl));
      vint16m2_t  q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8t, vl), vl);
      suml2 = __riscv_vmul_vv_i16m2(tr0, q80, vl);
      uint8_t p = 3;
      for (int t = 0; t < 4; ++t) {
        vuint16m2_t tqn = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tqb, p, vl), 3, vl), 8, vl);
        vint16m2_t  trn = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tqn, 1, vl));
        vint16m2_t  q8n = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8t + (t + 1) * 16, vl), vl);
        suml2 = __riscv_vmacc_vv_i16m2(suml2, trn, q8n, vl);
        p *= 3;
      }
    }

    /* ---- (C) qh region: SINGLE 16-lane pass, i16m2 -> q8[240..255] ----
     * broadcast the 4 qh bytes across a 16-byte register, lane(4l+j)=qh[j],
     * multiply by pow16 so lane(4l+j) = qh[j]*pow3[l] (l=0..3, j=0..3). */
    vint16m2_t suml3;
    {
      const size_t vl = 16;
      uint32_t qh; { const uint8_t *p = tq + 48; qh = (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }
      vuint8m1_t tqb = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vmv_v_x_u32m1(qh, vl / 4));
      vuint8m1_t pw  = __riscv_vle8_v_u8m1(pow16, vl);
      vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vv_u8m1(tqb, pw, vl), 3, vl), 8, vl);
      vint16m2_t  tr0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl));
      vint16m2_t  q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 240, vl), vl);
      suml3 = __riscv_vmul_vv_i16m2(tr0, q80, vl);
    }

    /* ---- (D) fold the three accumulators + ONE vwredsum ---- */
    vint16m2_t sumb = __riscv_vadd_vv_i16m2(__riscv_vget_v_i16m4_i16m2(suml1, 0),
                                            __riscv_vget_v_i16m4_i16m2(suml1, 1), 16);
    sumb = __riscv_vadd_vv_i16m2(sumb, suml2, 16);
    sumb = __riscv_vadd_vv_i16m2(sumb, suml3, 16);
    vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
    int sumi = __riscv_vmv_x_s_i32m1_i32(red);

    /* ---- (E) single-scale scalar fp32 fold (byte-exact statement order to _generic) ---- */
    float dx = (float)*(const _Float16 *)(xb + 52);
    float dy = *(const float *)(yb + 0);
    sumf += (float)sumi * (dx * dy);
  }

  *s = sumf;
}
