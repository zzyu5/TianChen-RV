// ggml's SHIPPED RVV tq1_0_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch path).
// Lifted verbatim from llama.cpp quants.c:6074, raw byte offsets.
// block_tq1_0=54B: qs[48]@0 qh[4]@48 d@52(fp16). Base-3-packed ternary {-1,0,+1}.
// block_q8_K=292B: d@0(float32) qs[256]@4 bsums[16]@260. bsums UNUSED.
// _vl256: first loop e8m1 vl=32 (i16m2), second loop e8mf2 vl=16 (i16m1), third loop
//   qh base-3 vl=16 (i16m1) w/ pow[16]. Fold: sumf += sumi*(y.d_f32*fp16(x.d)).
// FOLD ASSOCIATION: ggml's tq1_0 folds (sumi*y.d)*x.d -> small nonzero rel-norm vs
//   OURS (sumi*(y.d*x.d)); matched-assoc ref collapses to 0.0 (same as RVV).
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define QK_K 256

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int WB = 54, YB = 292;
  const int nb = (int)n / QK_K;
  float sumf = 0.0f;
  uint8_t pow[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};
  for (int i = 0; i < nb; i++) {
    const uint8_t *xb = vx + (size_t)i * WB, *yb = vy + (size_t)i * YB;
    const uint8_t *xqs = xb + 0;                   // x.qs[48]
    const uint8_t *xqh = xb + 48;                  // x.qh[4]
    const int8_t  *q8  = (const int8_t *)(yb + 4); // y.qs[256]

    // First loop (vl=32, m1->i16m2).
    vint16m2_t suml1;
    {
      const int vl = 32;
      vuint8m1_t tq = __riscv_vle8_v_u8m1(xqs + 0, vl);

      vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tq, 3, vl), 8, vl);
      vuint16m2_t tq1 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 3, vl), 3, vl), 8, vl);
      vuint16m2_t tq2 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 9, vl), 3, vl), 8, vl);
      vuint16m2_t tq3 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 27, vl), 3, vl), 8, vl);
      vuint16m2_t tq4 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 81, vl), 3, vl), 8, vl);

      vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 0, vl), vl);
      vint16m2_t q81 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 32, vl), vl);
      vint16m2_t q82 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 64, vl), vl);
      vint16m2_t q83 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 96, vl), vl);
      vint16m2_t q84 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8 + 128, vl), vl);

      vint16m2_t sum0 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
      vint16m2_t sum1 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq1, 1, vl)), q81, vl);
      vint16m2_t sum2 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq2, 1, vl)), q82, vl);
      vint16m2_t sum3 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq3, 1, vl)), q83, vl);
      vint16m2_t sum4 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq4, 1, vl)), q84, vl);

      vint16m2_t sumi0 = __riscv_vadd_vv_i16m2(sum0, sum1, vl);
      vint16m2_t sumi1 = __riscv_vadd_vv_i16m2(sum2, sum3, vl);
      suml1 = __riscv_vadd_vv_i16m2(sum4, __riscv_vadd_vv_i16m2(sumi0, sumi1, vl), vl);
    }

    // Second loop (vl=16, mf2->i16m1).
    vint16m1_t suml2;
    {
      const int vl = 16;
      vuint8mf2_t tq = __riscv_vle8_v_u8mf2(xqs + 32, vl);

      vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(tq, 3 * 1, vl), 8, vl);
      vuint16m1_t tq1 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 3, vl), 3, vl), 8, vl);
      vuint16m1_t tq2 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 9, vl), 3, vl), 8, vl);
      vuint16m1_t tq3 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 27, vl), 3, vl), 8, vl);
      vuint16m1_t tq4 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 81, vl), 3, vl), 8, vl);

      vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 160, vl), vl);
      vint16m1_t q81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 176, vl), vl);
      vint16m1_t q82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 192, vl), vl);
      vint16m1_t q83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 208, vl), vl);
      vint16m1_t q84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 224, vl), vl);

      vint16m1_t sum0 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
      vint16m1_t sum1 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq1, 1, vl)), q81, vl);
      vint16m1_t sum2 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq2, 1, vl)), q82, vl);
      vint16m1_t sum3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq3, 1, vl)), q83, vl);
      vint16m1_t sum4 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq4, 1, vl)), q84, vl);

      vint16m1_t sumi0 = __riscv_vadd_vv_i16m1(sum0, sum1, vl);
      vint16m1_t sumi1 = __riscv_vadd_vv_i16m1(sum2, sum3, vl);
      suml2 = __riscv_vadd_vv_i16m1(sum4, __riscv_vadd_vv_i16m1(sumi0, sumi1, vl), vl);
    }

    // Third loop (qh base-3, vl=16, i16m1).
    vint16m1_t suml3;
    {
      const int vl = 16;
      uint32_t qh;
      memcpy(&qh, xqh + 0, 4);
      __asm__ __volatile__("" : "+r"(qh));
      vuint8mf2_t tq = __riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(qh, vl / 4));
      vuint8mf2_t p = __riscv_vle8_v_u8mf2(pow, vl);
      vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vv_u8mf2(tq, p, vl), 3, vl), 8, vl);
      vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(q8 + 240, vl), vl);
      suml3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
    }

    vint16m1_t sumb = __riscv_vadd_vv_i16m1(__riscv_vget_v_i16m2_i16m1(suml1, 0), __riscv_vget_v_i16m2_i16m1(suml1, 1), 16);
    sumb = __riscv_vadd_vv_i16m1(sumb, __riscv_vadd_vv_i16m1(suml2, suml3, 16), 16);

    vint32m1_t sum = __riscv_vwredsum_vs_i16m1_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 52, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    sumf += __riscv_vmv_x_s_i32m1_i32(sum) * yd * (float)xd;
  }
  *s = sumf;
}
