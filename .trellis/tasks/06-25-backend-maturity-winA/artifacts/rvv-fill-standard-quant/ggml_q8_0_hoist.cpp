// ggml q8_0 ref, IDENTICAL body but with v_zero hoisted out of the block loop
// (advisor discriminator: is our 1.19× just a loop-invariant clang didn't lift
// for ggml's single-expression form?). Same symbol kern_ggml.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 32; const int WB = 34, YB = 34;
  const int nb = (int)n / qk; float sumf = 0; size_t vl = qk;
  vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);   // HOISTED
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * WB, *yb = vy + (size_t)ib * YB;
    _Float16 xf, yf; memcpy(&xf, xb + 0, 2); memcpy(&yf, yb + 0, 2);
    const int8_t *xqs = (const int8_t *)(xb + 2), *yqs = (const int8_t *)(yb + 2);
    vint8m2_t bx_0 = __riscv_vle8_v_i8m2(xqs, vl);
    vint8m2_t by_0 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);
    vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);
    sumf += sumi * ((float)xf * (float)yf);
  }
  *s = sumf;
}
