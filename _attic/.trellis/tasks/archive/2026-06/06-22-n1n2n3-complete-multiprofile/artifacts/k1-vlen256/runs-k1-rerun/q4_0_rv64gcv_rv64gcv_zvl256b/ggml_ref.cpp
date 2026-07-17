#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 16;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * 18, *yb = vy + (size_t)ib * 34;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
    float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
    sumf += sumi * dx * dy;
  }
  *s = sumf;
}
