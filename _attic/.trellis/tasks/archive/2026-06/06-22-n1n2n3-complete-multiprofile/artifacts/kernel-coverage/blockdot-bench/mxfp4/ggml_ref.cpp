// ggml's SHIPPED RVV mxfp4_q8_0 kernel, _vl128 variant, lifted verbatim from
// llama.cpp ggml-cpu/arch/riscv/quants.c:6470. block_mxfp4=17B: e@0(E8M0 byte) qs[16]@1.
// block_q8_0=34B: d@0(fp16) qs[32]@2. FP4 e2m1 codebook gather (kvalues_mxfp4[16]).
// NOTE: ggml's vl128 path processes 2 blocks/iter (ib+1<nb) and DROPS the last
// block on odd nb. The harness keeps nb even so this is moot.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

static const int8_t kvalues_mxfp4[16] = {
  0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ggml_e8m0_to_fp32_half (ggml-impl.h): value*0.5 to match kvalues=2*E2M1.
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
  const vint8m2_t values = __riscv_vle8_v_i8m2(kvalues_mxfp4, 16);
  int acc1, acc2;
  for (; ib + 1 < nb; ib += 2) {
    const uint8_t *x0 = vx + (size_t)(ib + 0) * WB, *x1 = vx + (size_t)(ib + 1) * WB;
    const uint8_t *y0 = vy + (size_t)(ib + 0) * YB, *y1 = vy + (size_t)(ib + 1) * YB;
    vuint8m1_t mx_packed1 = __riscv_vle8_v_u8m1(x0 + 1, 16);
    vint8m2_t  q8b1 = __riscv_vle8_v_i8m2((const int8_t *)(y0 + 2), 32);
    vuint8m1_t mx_packed2 = __riscv_vle8_v_u8m1(x1 + 1, 16);
    vint8m2_t  q8b2 = __riscv_vle8_v_i8m2((const int8_t *)(y1 + 2), 32);
    vuint8m2_t mxbits1 = __riscv_vcreate_v_u8m1_u8m2(
        __riscv_vand_vx_u8m1(mx_packed1, 0xf, 16),
        __riscv_vsrl_vx_u8m1(mx_packed1, 4, 16));
    vuint8m2_t mxbits2 = __riscv_vcreate_v_u8m1_u8m2(
        __riscv_vand_vx_u8m1(mx_packed2, 0xf, 16),
        __riscv_vsrl_vx_u8m1(mx_packed2, 4, 16));
    vint8m2_t mxb1 = __riscv_vrgather_vv_i8m2(values, mxbits1, 32);
    vint8m2_t mxb2 = __riscv_vrgather_vv_i8m2(values, mxbits2, 32);
    vint16m4_t sum1 = __riscv_vwmul_vv_i16m4(q8b1, mxb1, 32);
    vint16m4_t sum2 = __riscv_vwmul_vv_i16m4(q8b2, mxb2, 32);
    __riscv_vse32_v_i32m1(&acc1, __riscv_vwredsum_vs_i16m4_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    __riscv_vse32_v_i32m1(&acc2, __riscv_vwredsum_vs_i16m4_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    uint8_t xe1 = x0[0], xe2 = x1[0];
    _Float16 yd1, yd2; uint16_t t;
    memcpy(&t, y0 + 0, 2); memcpy(&yd1, &t, 2);
    memcpy(&t, y1 + 0, 2); memcpy(&yd2, &t, 2);
    sumf += (e8m0_to_fp32_half(xe1) * (float)yd1 * acc1);
    sumf += (e8m0_to_fp32_half(xe2) * (float)yd2 * acc2);
  }
  *s = sumf;
}
