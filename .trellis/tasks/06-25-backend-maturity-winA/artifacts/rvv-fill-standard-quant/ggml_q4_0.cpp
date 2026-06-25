// ggml's SHIPPED RVV ggml_vec_dot_q4_0_q8_0 kernel lifted verbatim from
// llama.cpp ggml/src/ggml-cpu/arch/riscv/quants.c (vl-agnostic VLMAX path,
// taken on VLEN128). block_q4_0=18B (d@0, qs[16]@2); block_q8_0=34B (d@0,
// qs[32]@2). QK=32. Wrapped in the 4-arg byte-offset ABI adapter.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 32;
  const int WB = 18, YB = 34;
  const int nb = (int)n / qk;
  float sumf = 0;
  size_t vl = qk / 2;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * WB;
    const uint8_t *yb = vy + (size_t)ib * YB;
    _Float16 xf, yf; memcpy(&xf, xb + 0, 2); memcpy(&yf, yb + 0, 2);
    const uint8_t *xqs = xb + 2;
    const int8_t  *yqs = (const int8_t *)(yb + 2);
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yqs, vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1(yqs + 16, vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);
    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    sumf += sumi * (float)xf * (float)yf;
  }
  *s = sumf;
}
