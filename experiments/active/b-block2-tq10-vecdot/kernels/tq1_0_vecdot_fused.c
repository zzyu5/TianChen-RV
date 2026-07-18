/* tq1_0_vecdot_fused.c -- OWNED FUSED ternary vec_dot leaf: tq1_0 x q8_K (M=1).
 *
 * B线第二块 P1 (公式墙攻坚). Reproduces the ggml opponent's readable C-intrinsic
 * structure (ggml_vec_dot_tq1_0_q8_K_vl128 / _vl256, tier=hand-tuned, gather=0):
 *   base-3 powers-of-3 unpack (vmul.vx / vwmulu.vx x3 / vsrl >>8) -> trit {-1,0,1},
 *   FUSED directly into a WIDE i16 accumulator via vwmacc (trit * q8) -- NO aux8
 *   scratch round-trip -- then ONE final vwredsum per super-block.
 *
 * This is the FUSED leaf that closes the two structural gaps of the current emitter
 * leaf (emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10, RVVToEmitCTernaryBinary.cpp):
 *   (1) the aux8[256] scratch store+reload (512 mem ops / super-block) -> eliminated,
 *   (2) the 8x vwredsum per super-block -> collapsed to 1 (matches opponent reduce=1).
 *
 * PURE C-INTRINSIC. NO inline-asm, NO pinned schedule -- the opponent is C-intrinsic
 * (clang131 ~= gcc132 per objdump-recon), so an owned C-intrinsic is on-position.
 *
 * BYTE-EXACT integer core to ggml_vec_dot_tq1_0_q8_K_generic by construction:
 *   identical pow3[] mod-256 decode, identical q8 index map, order-free integer sum.
 *   The single fp32 fold `sumf += (float)sum * (fp16(x.d) * y.d)` matches _generic's
 *   exact statement order (no fp reassociation).
 *
 * block_tq1_0 (54B): qs[48] @0, qh[4] @48, d(fp16) @52.  QK_K = 256.
 * block_q8_K (292B): d(f32) @0, qs[256] @4, bsums[16] @260.
 *
 * Requires -march with zfh (hardware fcvt.s.h for the fp16 scale read), VLEN >= 128.
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
  static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
  float sumf = 0.0f;

  for (size_t ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + ib * 54;         /* block_tq1_0 base */
    const uint8_t *yb = vy + ib * 292;        /* block_q8_K  base */
    const int8_t  *q8 = (const int8_t *)(yb + 4);

    /* ONE wide i16 accumulator (VLMAX(e16m4) >= 32 for VLEN >= 128). 32 active
     * lanes are zeroed; every trit*q8 product is folded into some lane in [0,32)
     * (order-free integer sum). Per-lane bound: <=14 products, |trit|<=1,
     * |q8|<=127 => |lane| <= 14*127 = 1778 << 32767, so i16 never overflows. */
    size_t vl32 = __riscv_vsetvl_e16m4(32);
    vint16m4_t vacc = __riscv_vmv_v_x_i16m4(0, vl32);

    /* (A) main qs region: j=0, l=0..4, m=0..31; q8 index = l*32 + m (0..159).
     * The 32 weight bytes qs[0..31] are loaded ONCE and reused across the 5 digit
     * planes (only the pow3[l] scalar differs). */
    {
      size_t vl = __riscv_vsetvl_e8m2(32);
      vuint8m2_t w = __riscv_vle8_v_u8m2(xb + 0, vl);
      for (int l = 0; l < 5; ++l) {
        vuint8m2_t  q   = __riscv_vmul_vx_u8m2(w, pow3[l], vl);   /* (u8)(w*pow3) */
        vuint16m4_t q3  = __riscv_vwmulu_vx_u16m4(q, 3, vl);      /* widen *3 */
        vuint16m4_t xiw = __riscv_vsrl_vx_u16m4(q3, 8, vl);       /* >>8 in {0,1,2} */
        vuint8m2_t  xin = __riscv_vncvt_x_x_w_u8m2(xiw, vl);      /* narrow (lossless) */
        vint8m2_t   xii = __riscv_vreinterpret_v_u8m2_i8m2(xin);
        vint8m2_t   tr  = __riscv_vadd_vx_i8m2(xii, -1, vl);      /* trit {-1,0,1} */
        vint8m2_t   q8v = __riscv_vle8_v_i8m2(q8 + l * 32, vl);
        vacc = __riscv_vwmacc_vv_i16m4(vacc, tr, q8v, vl);        /* vacc += trit*q8 */
      }
    }

    /* (B) tail qs region: j=32, l=0..4, m=0..15; q8 index = 160 + l*16 + m. */
    {
      size_t vl = __riscv_vsetvl_e8m2(16);
      vuint8m2_t w = __riscv_vle8_v_u8m2(xb + 32, vl);
      for (int l = 0; l < 5; ++l) {
        vuint8m2_t  q   = __riscv_vmul_vx_u8m2(w, pow3[l], vl);
        vuint16m4_t q3  = __riscv_vwmulu_vx_u16m4(q, 3, vl);
        vuint16m4_t xiw = __riscv_vsrl_vx_u16m4(q3, 8, vl);
        vuint8m2_t  xin = __riscv_vncvt_x_x_w_u8m2(xiw, vl);
        vint8m2_t   xii = __riscv_vreinterpret_v_u8m2_i8m2(xin);
        vint8m2_t   tr  = __riscv_vadd_vx_i8m2(xii, -1, vl);
        vint8m2_t   q8v = __riscv_vle8_v_i8m2(q8 + 160 + l * 16, vl);
        vacc = __riscv_vwmacc_vv_i16m4(vacc, tr, q8v, vl);
      }
    }

    /* (C) qh region: l=0..3, j=0..3; weight byte qh[j], q8 index = 240 + l*4 + j. */
    {
      size_t vl = __riscv_vsetvl_e8m2(4);
      vuint8m2_t w = __riscv_vle8_v_u8m2(xb + 48, vl);
      for (int l = 0; l < 4; ++l) {
        vuint8m2_t  q   = __riscv_vmul_vx_u8m2(w, pow3[l], vl);
        vuint16m4_t q3  = __riscv_vwmulu_vx_u16m4(q, 3, vl);
        vuint16m4_t xiw = __riscv_vsrl_vx_u16m4(q3, 8, vl);
        vuint8m2_t  xin = __riscv_vncvt_x_x_w_u8m2(xiw, vl);
        vint8m2_t   xii = __riscv_vreinterpret_v_u8m2_i8m2(xin);
        vint8m2_t   tr  = __riscv_vadd_vx_i8m2(xii, -1, vl);
        vint8m2_t   q8v = __riscv_vle8_v_i8m2(q8 + 240 + l * 4, vl);
        vacc = __riscv_vwmacc_vv_i16m4(vacc, tr, q8v, vl);
      }
    }

    /* (D) ONE reduction over the 32-lane i16 accumulator -> integer super-block dot. */
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t red  = __riscv_vwredsum_vs_i16m4_i32m1(vacc, zero, vl32);
    int sumi = __riscv_vmv_x_s_i32m1_i32(red);

    /* (E) single-scale scalar fp32 fold (byte-exact statement order to _generic). */
    float dx = (float)*(const _Float16 *)(xb + 52);   /* fp16 tq1_0 scale */
    float dy = *(const float *)(yb + 0);              /* f32 q8_K scale */
    sumf += (float)sumi * (dx * dy);
  }

  *s = sumf;
}
