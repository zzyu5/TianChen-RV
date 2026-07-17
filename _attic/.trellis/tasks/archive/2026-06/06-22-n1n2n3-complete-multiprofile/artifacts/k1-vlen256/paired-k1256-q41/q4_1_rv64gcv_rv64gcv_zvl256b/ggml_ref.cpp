#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 16;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xq = vx + ib * 20 + 4; const int8_t *yq = (const int8_t *)(vy + ib * 36 + 4);
    float xd = (float)*(const _Float16 *)(vx + ib * 20), xm = (float)*(const _Float16 *)(vx + ib * 20 + 2);
    float yd = (float)*(const _Float16 *)(vy + ib * 36), ys = (float)*(const _Float16 *)(vy + ib * 36 + 2);
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xq, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yq, vl), y1 = __riscv_vle8_v_i8m1(yq + 16, vl);
    vuint8m1_t a = __riscv_vand_vx_u8m1(tx, 0x0F, vl), l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(a), v1 = __riscv_vreinterpret_v_u8m1_i8m1(l);
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl), m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl), r = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += (xd * yd) * sumi + xm * ys;
  }
  *s = sumf;
}
