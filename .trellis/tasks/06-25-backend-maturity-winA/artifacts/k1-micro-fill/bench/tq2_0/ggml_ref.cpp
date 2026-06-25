// ggml's SHIPPED RVV tq2_0_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch path).
// Lifted verbatim from llama.cpp quants.c:6382, raw byte offsets.
// block_tq2_0=66B: qs[64]@0 d@64(fp16). block_q8_K=292B: d@0(float32) qs[256]@4
// bsums[16]@260(int16). 2-bit ternary {-1,0,+1}. Fold: sumf += sumi*(y.d_f32*fp16(x.d)).
// _vl256 uses e8m1 (32 lanes at VLEN256 = full m1 register), vwmacc_i16m2.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define QK_K 256

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int WB = 66, YB = 292;
  const int nb = (int)n / QK_K;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = vx + (size_t)i * WB, *yb = vy + (size_t)i * YB;
    const uint8_t *xqs = xb + 0;
    const int8_t  *yqs = (const int8_t *)(yb + 4);
    int32_t sumi = 0;
    for (size_t j = 0; j < 64 /* sizeof qs */; j += 32) {
      const int8_t *py0 = &yqs[j * 4 + 0 * 32];
      const int8_t *py1 = &yqs[j * 4 + 1 * 32];
      const int8_t *py2 = &yqs[j * 4 + 2 * 32];
      const int8_t *py3 = &yqs[j * 4 + 3 * 32];
      const uint8_t *px = &xqs[j];

      size_t vlmax_16m2 = __riscv_vsetvl_e16m2(32);
      vint16m2_t vacc16 = __riscv_vmv_v_x_i16m2(0, vlmax_16m2);

      size_t vl = __riscv_vsetvl_e8m1(32);
      vuint8m1_t vx_u8 = __riscv_vle8_v_u8m1(px, vl);
      vint8m1_t vy0 = __riscv_vle8_v_i8m1(py0, vl);
      vint8m1_t vy1 = __riscv_vle8_v_i8m1(py1, vl);
      vint8m1_t vy2 = __riscv_vle8_v_i8m1(py2, vl);
      vint8m1_t vy3 = __riscv_vle8_v_i8m1(py3, vl);

      vuint8m1_t t0 = __riscv_vand_vx_u8m1(vx_u8, 0x03, vl);
      vint8m1_t vq0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t0), 1, vl);
      vuint8m1_t t1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 2, vl), 0x03, vl);
      vint8m1_t vq1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t1), 1, vl);
      vuint8m1_t t2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 4, vl), 0x03, vl);
      vint8m1_t vq2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t2), 1, vl);
      vuint8m1_t t3 = __riscv_vsrl_vx_u8m1(vx_u8, 6, vl);
      vint8m1_t vq3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t3), 1, vl);

      vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq0, vy0, vl);
      vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq1, vy1, vl);
      vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq2, vy2, vl);
      vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq3, vy3, vl);

      vlmax_16m2 = __riscv_vsetvl_e16m2(32);
      vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vred32 = __riscv_vwredsum_vs_i16m2_i32m1(vacc16, vzero32, vlmax_16m2);
      sumi += __riscv_vmv_x_s_i32m1_i32(vred32);
    }
    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 64, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    const float d = yd * (float)xd;
    sumf += (float)sumi * d;
  }
  *s = sumf;
}
