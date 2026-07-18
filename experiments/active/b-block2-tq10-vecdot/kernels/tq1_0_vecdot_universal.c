/* tq1_0_vecdot_universal.c -- OWNED VLEN-UNIVERSAL LEAN ternary vec_dot leaf.
 *
 * B线第二块 P1. v3: combines
 *   - v1's VLEN-UNIVERSAL single i16m4 accumulator + ONE vwredsum over a FIXED vl
 *     (works byte-exact at BOTH VLEN128 and VLEN256 -- unlike the ggml _vl128 form,
 *      whose vget-split fold is VLEN128-hardcoded and needs a separate _vl256), and
 *   - v2's LEAN per-plane unpack adopted from the ggml hand-tuned readable structure:
 *       digit-0 skips vmul_vx(*1); trit kept in i16 via vsub_vx_u16(tq,1)+reinterpret
 *       (no narrow); q8 pre-widened to i16 (vwcvt) with vmul_vv(init)/vmacc_vv chain;
 *       qh decoded in a SINGLE pass via a broadcast + pow16 lookup.
 * All regions accumulate into ONE i16m4 vacc at a FIXED vl (32 main / 16 tail+qh),
 * so m4/m2 VLMAX >= the vl for any VLEN >= 128 -> byte-exact on rvv AND k1. ONE reduce.
 *
 * PURE C-INTRINSIC. NO inline-asm, NO pinned schedule. Byte-exact integer core to
 * ggml_vec_dot_tq1_0_q8_K_generic (identical pow3 mod-256 decode + q8 index map,
 * order-free integer sum; i16 lanes bound <= 11*127 = 1397 << 32767, no overflow).
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
    const uint8_t *tq = xb;
    const int8_t  *q8 = (const int8_t *)(yb + 4);

    vint16m4_t vacc;   /* ONE accumulator; lanes 0..31 hold the whole super-block dot */

    /* ---- (A) main qs: 32 lanes, digits 0..4 -> q8[0..159] ---- */
    {
      const size_t vl = 32;
      vuint8m2_t tqb = __riscv_vle8_v_u8m2(tq, vl);
      vuint16m4_t tq0 = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(tqb, 3, vl), 8, vl); /* digit0 skips *1 */
      vint16m4_t  tr0 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tq0, 1, vl));
      vint16m4_t  q80 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
      vacc = __riscv_vmul_vv_i16m4(tr0, q80, vl);   /* init accumulator (lanes 0..31) */
      uint8_t p = 3;
      for (int t = 0; t < 4; ++t) {
        vuint16m4_t tqn = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vx_u8m2(tqb, p, vl), 3, vl), 8, vl);
        vint16m4_t  trn = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tqn, 1, vl));
        vint16m4_t  q8n = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8 + (t + 1) * 32, vl), vl);
        vacc = __riscv_vmacc_vv_i16m4(vacc, trn, q8n, vl);
        p *= 3;
      }
    }

    /* ---- (B) tail qs: 16 lanes, digits 0..4 -> q8[160..239] (into vacc[0..15]) ---- */
    {
      const size_t vl = 16;
      vuint8m2_t tqb = __riscv_vle8_v_u8m2(tq + 32, vl);
      const int8_t *q8t = q8 + 160;
      uint8_t p = 1;
      for (int l = 0; l < 5; ++l) {
        vuint8m2_t tqx = (l == 0) ? tqb : __riscv_vmul_vx_u8m2(tqb, p, vl);
        vuint16m4_t tql = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(tqx, 3, vl), 8, vl);
        vint16m4_t  trl = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tql, 1, vl));
        vint16m4_t  q8l = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8t + l * 16, vl), vl);
        vacc = __riscv_vmacc_vv_i16m4(vacc, trl, q8l, vl);
        p *= 3;
      }
    }

    /* ---- (C) qh: SINGLE pass, 16 lanes (4 planes x 4) -> q8[240..255] (into vacc[0..15]) ---- */
    {
      const size_t vl = 16;
      uint32_t qh; { const uint8_t *p = tq + 48; qh = (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }
      vuint8m2_t tqb = __riscv_vreinterpret_v_u32m2_u8m2(__riscv_vmv_v_x_u32m2(qh, 4)); /* 4 u32 lanes = 16 bytes = qh x4 */
      vuint8m2_t pw  = __riscv_vle8_v_u8m2(pow16, vl);
      vuint16m4_t tqh = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vv_u8m2(tqb, pw, vl), 3, vl), 8, vl);
      vint16m4_t  trh = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tqh, 1, vl));
      vint16m4_t  q8h = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8 + 240, vl), vl);
      vacc = __riscv_vmacc_vv_i16m4(vacc, trh, q8h, vl);
    }

    /* ---- (D) ONE reduce over the 32 active lanes (fixed vl -> VLEN-universal) ---- */
    vint32m1_t red = __riscv_vwredsum_vs_i16m4_i32m1(vacc, __riscv_vmv_v_x_i32m1(0, 1), 32);
    int sumi = __riscv_vmv_x_s_i32m1_i32(red);

    /* ---- (E) single-scale scalar fp32 fold (byte-exact statement order to _generic) ---- */
    float dx = (float)*(const _Float16 *)(xb + 52);
    float dy = *(const float *)(yb + 0);
    sumf += (float)sumi * (dx * dy);
  }

  *s = sumf;
}
