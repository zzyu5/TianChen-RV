// ggml's SHIPPED RVV iq4_nl_q8_0 kernel, _vl128 variant, lifted verbatim from
// llama.cpp quants.c:5479, raw byte offsets. block_iq4_nl=18B: d@0(fp16) qs[16]@2.
// block_q8_0=34B: d@0(fp16) qs[32]@2. FP4 codebook gather (kvalues_iq4nl[16]).
// NOTE: ggml's vl128 path processes 2 blocks/iter (ib+1<nb) and DROPS the last
// block on odd nb. The harness keeps nb even so this is moot.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

static const int8_t kvalues_iq4nl[16] = {
  -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int QK = 32; const int WB = 18, YB = 34;
  const int nb = (int)n / QK;
  int ib = 0; float sumf = 0;
  const vint8m2_t values = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
  int acc1, acc2;
  for (; ib + 1 < nb; ib += 2) {
    const uint8_t *x0 = vx + (size_t)(ib + 0) * WB, *x1 = vx + (size_t)(ib + 1) * WB;
    const uint8_t *y0 = vy + (size_t)(ib + 0) * YB, *y1 = vy + (size_t)(ib + 1) * YB;
    vuint8m1_t iq4_packed1 = __riscv_vle8_v_u8m1(x0 + 2, 16);
    vint8m2_t  q8b1 = __riscv_vle8_v_i8m2((const int8_t *)(y0 + 2), 32);
    vuint8m1_t iq4_packed2 = __riscv_vle8_v_u8m1(x1 + 2, 16);
    vint8m2_t  q8b2 = __riscv_vle8_v_i8m2((const int8_t *)(y1 + 2), 32);
    vuint8m2_t iq4bits1 = __riscv_vcreate_v_u8m1_u8m2(
        __riscv_vand_vx_u8m1(iq4_packed1, 0xf, 16),
        __riscv_vsrl_vx_u8m1(iq4_packed1, 4, 16));
    vuint8m2_t iq4bits2 = __riscv_vcreate_v_u8m1_u8m2(
        __riscv_vand_vx_u8m1(iq4_packed2, 0xf, 16),
        __riscv_vsrl_vx_u8m1(iq4_packed2, 4, 16));
    vint8m2_t iq4b1 = __riscv_vrgather_vv_i8m2(values, iq4bits1, 32);
    vint8m2_t iq4b2 = __riscv_vrgather_vv_i8m2(values, iq4bits2, 32);
    vint16m4_t sum1 = __riscv_vwmul_vv_i16m4(q8b1, iq4b1, 32);
    vint16m4_t sum2 = __riscv_vwmul_vv_i16m4(q8b2, iq4b2, 32);
    __riscv_vse32_v_i32m1(&acc1, __riscv_vwredsum_vs_i16m4_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    __riscv_vse32_v_i32m1(&acc2, __riscv_vwredsum_vs_i16m4_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    _Float16 xd1, yd1, xd2, yd2; uint16_t t;
    memcpy(&t, x0 + 0, 2); memcpy(&xd1, &t, 2); memcpy(&t, y0 + 0, 2); memcpy(&yd1, &t, 2);
    memcpy(&t, x1 + 0, 2); memcpy(&xd2, &t, 2); memcpy(&t, y1 + 0, 2); memcpy(&yd2, &t, 2);
    sumf += ((float)xd1 * (float)yd1 * acc1);
    sumf += ((float)xd2 * (float)yd2 * acc2);
  }
  *s = sumf;
}
