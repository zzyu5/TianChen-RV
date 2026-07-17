// ggml's SHIPPED RVV tq2_0_q8_K kernel, _vl128 variant, lifted verbatim from
// llama.cpp quants.c:6300, raw byte offsets. block_tq2_0=66B: qs[64]@0 d@64(fp16).
// block_q8_K=292B: d@0(float32) qs[256]@4 bsums[16]@260(int16). 2-bit ternary
// {-1,0,+1}. Fold: sumf += sumi * (y.d_float32 * fp16(x.d)). bsums UNUSED here.
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
      size_t vl = __riscv_vsetvl_e16m4(32);
      vint16m4_t vacc16 = __riscv_vmv_v_x_i16m4(0, vl);
      vl = __riscv_vsetvl_e8m2(32);
      vuint8m2_t vx_u8 = __riscv_vle8_v_u8m2(px, vl);
      { vuint8m2_t t0 = __riscv_vand_vx_u8m2(vx_u8, 0x03, vl);
        vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t0), 1, vl);
        vint8m2_t vy = __riscv_vle8_v_i8m2(py0, vl);
        vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl); }
      __asm__ volatile("" ::: "memory");
      { vuint8m2_t t1 = __riscv_vsrl_vx_u8m2(vx_u8, 2, vl); t1 = __riscv_vand_vx_u8m2(t1, 0x03, vl);
        vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t1), 1, vl);
        vint8m2_t vy = __riscv_vle8_v_i8m2(py1, vl);
        vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl); }
      __asm__ volatile("" ::: "memory");
      { vuint8m2_t t2 = __riscv_vsrl_vx_u8m2(vx_u8, 4, vl); t2 = __riscv_vand_vx_u8m2(t2, 0x03, vl);
        vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t2), 1, vl);
        vint8m2_t vy = __riscv_vle8_v_i8m2(py2, vl);
        vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl); }
      __asm__ volatile("" ::: "memory");
      { vuint8m2_t t3 = __riscv_vsrl_vx_u8m2(vx_u8, 6, vl);
        vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t3), 1, vl);
        vint8m2_t vy = __riscv_vle8_v_i8m2(py3, vl);
        vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl); }
      __asm__ volatile("" ::: "memory");
      vl = __riscv_vsetvl_e16m4(32);
      vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vred32 = __riscv_vwredsum_vs_i16m4_i32m1(vacc16, vzero32, vl);
      sumi += __riscv_vmv_x_s_i32m1_i32(vred32);
    }
    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 64, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    const float d = yd * (float)xd;
    sumf += (float)sumi * d;
  }
  *s = sumf;
}
