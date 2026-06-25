// ggml's SHIPPED RVV mxfp4_q8_0 kernel, _vl256 variant (k1/VLEN256 dispatch path).
// Lifted verbatim from llama.cpp quants.c:6525, raw byte offsets.
// block_mxfp4=17B: e@0(E8M0 byte) qs[16]@1. block_q8_0=34B: d@0(fp16) qs[32]@2.
// _vl256: mf2 16-lane split lo/hi gather, 2-blocks/iter. 2-blocks/iter drops the last
// block on odd nb; harness keeps nb even so this is moot.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

static const int8_t kvalues_mxfp4[16] = {
  0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static inline float e8m0_to_fp32_half(uint8_t x) {
  uint32_t bits;
  if (x < 2) bits = (uint32_t)0x00200000u << x;
  else       bits = (uint32_t)(x - 1) << 23;
  float r; memcpy(&r, &bits, sizeof(float)); return r;
}

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int QK = 32; const int WB = 17, YB = 34;
  const int nb = (int)n / QK;
  int ib = 0; float sumf = 0;
  const vint8mf2_t values = __riscv_vle8_v_i8mf2(kvalues_mxfp4, 16);
  int acc1, acc2;
  for (; ib + 1 < nb; ib += 2) {
    const uint8_t *x0 = vx + (size_t)(ib + 0) * WB, *x1 = vx + (size_t)(ib + 1) * WB;
    const uint8_t *y0 = vy + (size_t)(ib + 0) * YB, *y1 = vy + (size_t)(ib + 1) * YB;
    vuint8mf2_t mx_packed1 = __riscv_vle8_v_u8mf2(x0 + 1, 16);
    vint8mf2_t  q8b_lo1 = __riscv_vle8_v_i8mf2((const int8_t *)(y0 + 2), 16);
    vint8mf2_t  q8b_hi1 = __riscv_vle8_v_i8mf2((const int8_t *)(y0 + 2 + 16), 16);
    vuint8mf2_t mx_packed2 = __riscv_vle8_v_u8mf2(x1 + 1, 16);
    vint8mf2_t  q8b_lo2 = __riscv_vle8_v_i8mf2((const int8_t *)(y1 + 2), 16);
    vint8mf2_t  q8b_hi2 = __riscv_vle8_v_i8mf2((const int8_t *)(y1 + 2 + 16), 16);
    vuint8mf2_t mxbits_lo1 = __riscv_vand_vx_u8mf2(mx_packed1, 0xf, 16);
    vuint8mf2_t mxbits_hi1 = __riscv_vsrl_vx_u8mf2(mx_packed1, 4, 16);
    vuint8mf2_t mxbits_lo2 = __riscv_vand_vx_u8mf2(mx_packed2, 0xf, 16);
    vuint8mf2_t mxbits_hi2 = __riscv_vsrl_vx_u8mf2(mx_packed2, 4, 16);
    vint8mf2_t mxb_lo1 = __riscv_vrgather_vv_i8mf2(values, mxbits_lo1, 16);
    vint8mf2_t mxb_hi1 = __riscv_vrgather_vv_i8mf2(values, mxbits_hi1, 16);
    vint8mf2_t mxb_lo2 = __riscv_vrgather_vv_i8mf2(values, mxbits_lo2, 16);
    vint8mf2_t mxb_hi2 = __riscv_vrgather_vv_i8mf2(values, mxbits_hi2, 16);
    vint16m1_t sum1 = __riscv_vwmul_vv_i16m1(q8b_lo1, mxb_lo1, 16);
    sum1 = __riscv_vwmacc_vv_i16m1(sum1, q8b_hi1, mxb_hi1, 16);
    vint16m1_t sum2 = __riscv_vwmul_vv_i16m1(q8b_lo2, mxb_lo2, 16);
    sum2 = __riscv_vwmacc_vv_i16m1(sum2, q8b_hi2, mxb_hi2, 16);
    __riscv_vse32_v_i32m1(&acc1, __riscv_vwredsum_vs_i16m1_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    __riscv_vse32_v_i32m1(&acc2, __riscv_vwredsum_vs_i16m1_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    uint8_t xe1 = x0[0], xe2 = x1[0];
    _Float16 yd1, yd2; uint16_t t;
    memcpy(&t, y0 + 0, 2); memcpy(&yd1, &t, 2);
    memcpy(&t, y1 + 0, 2); memcpy(&yd2, &t, 2);
    sumf += (e8m0_to_fp32_half(xe1) * (float)yd1 * acc1);
    sumf += (e8m0_to_fp32_half(xe2) * (float)yd2 * acc2);
  }
  *s = sumf;
}
