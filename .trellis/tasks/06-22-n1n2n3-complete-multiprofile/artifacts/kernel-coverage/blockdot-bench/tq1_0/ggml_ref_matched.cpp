// ggml's SHIPPED RVV tq1_0_q8_K kernel, _vl128 variant, lifted verbatim from
// llama.cpp quants.c:5975, raw byte offsets. Base-3-packed ternary {-1,0,+1}.
// block_tq1_0=54B: qs[48]@0 qh[4]@48 d@52(fp16).
// block_q8_K=292B: d@0(float32) qs[256]@4 bsums[16]@260. bsums UNUSED.
// Fold: sumf += sumi * (y.d_float32 * fp16(x.d)).
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
    const uint8_t *tq = xb + 0;            // x.qs
    const uint8_t *xqh = xb + 48;          // x.qh[4]
    const int8_t  *q8 = (const int8_t *)(yb + 4); // y.qs

    // First loop.
    vint16m4_t suml1;
    {
      const int vl = 32;
      const vuint8m2_t tqb = __riscv_vle8_v_u8m2(tq, vl);
      tq += 32;
      {
        const vuint16m4_t tq0 = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(tqb, 3, vl), 8, vl);
        const vint16m4_t q80 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
        suml1 = __riscv_vmul_vv_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tq0, 1, vl)), q80, vl);
        q8 += 32;
      }
      uint8_t pow3 = 3;
      #pragma GCC unroll 1
      for (int t = 0; t < 4; t++) {
        const vuint16m4_t tqn = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vx_u8m2(tqb, pow3, vl), 3, vl), 8, vl);
        const vint16m4_t q8n = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
        suml1 = __riscv_vmacc_vv_i16m4(suml1, __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tqn, 1, vl)), q8n, vl);
        pow3 *= 3;
        q8 += 32;
      }
    }
    // Second loop.
    vint16m2_t suml2;
    {
      const int vl = 16;
      const vuint8m1_t tqb = __riscv_vle8_v_u8m1(tq, vl);
      {
        const vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tqb, 3, vl), 8, vl);
        const vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);
        suml2 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
        q8 += 16;
      }
      uint8_t pow3 = 3;
      #pragma GCC unroll 1
      for (int t = 0; t < 4; t++) {
        const vuint16m2_t tqn = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tqb, pow3, vl), 3, vl), 8, vl);
        const vint16m2_t q8n = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);
        suml2 = __riscv_vmacc_vv_i16m2(suml2, __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tqn, 1, vl)), q8n, vl);
        pow3 *= 3;
        q8 += 16;
      }
    }
    // Third loop.
    vint16m2_t suml3;
    {
      const int vl = 16;
      uint32_t qh;
      memcpy(&qh, xqh, 4);
      __asm__ __volatile__("" : "+r"(qh));
      const vuint8m1_t tqb = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vmv_v_x_u32m1(qh, vl / 4));
      const vuint8m1_t p = __riscv_vle8_v_u8m1(pow, vl);
      const vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vv_u8m1(tqb, p, vl), 3, vl), 8, vl);
      const vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);
      suml3 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
    }
    vint16m2_t sumb = __riscv_vadd_vv_i16m2(__riscv_vget_v_i16m4_i16m2(suml1, 0), __riscv_vget_v_i16m4_i16m2(suml1, 1), 16);
    sumb = __riscv_vadd_vv_i16m2(sumb, suml2, 16);
    sumb = __riscv_vadd_vv_i16m2(sumb, suml3, 16);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 52, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    sumf += (float)__riscv_vmv_x_s_i32m1_i32(sum) * (yd * (float)xd); // MATCHED assoc
  }
  *s = sumf;
}
