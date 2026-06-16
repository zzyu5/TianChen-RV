// ggml's REAL hand-written q4_1 x q8_1 RVV kernel (quants.c:277-326), as a
// SEPARATE TU with the SAME extern "C" signature + flags as the compiler-emitted
// shapes under test -- so the microbench has NO inline-vs-separate-TU asymmetry.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 20
#define Q8_STRIDE 36

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx,
                          const uint8_t *vy) {
  const int nb = (int)n / QK;
  float sumf = 0;
  size_t vl = QK / 2; // 16
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xq = vx + ib * Q4_STRIDE + 4;
    const int8_t *yq = (const int8_t *)(vy + ib * Q8_STRIDE + 4);
    uint16_t xd, xm, yd, ys;
    memcpy(&xd, vx + ib * Q4_STRIDE + 0, 2);
    memcpy(&xm, vx + ib * Q4_STRIDE + 2, 2);
    memcpy(&yd, vy + ib * Q8_STRIDE + 0, 2);
    memcpy(&ys, vy + ib * Q8_STRIDE + 2, 2);
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xq, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yq, vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1(yq + 16, vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += (fp16_to_fp32(xd) * fp16_to_fp32(yd)) * sumi +
            fp16_to_fp32(xm) * fp16_to_fp32(ys);
  }
  *s = sumf;
}
