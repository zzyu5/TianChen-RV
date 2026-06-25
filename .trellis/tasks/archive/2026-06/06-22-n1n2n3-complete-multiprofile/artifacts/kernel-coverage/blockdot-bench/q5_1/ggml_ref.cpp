// ggml's SHIPPED RVV q5_1_q8_1 kernel (unified __riscv_v body, vlenb==16 / VLEN128
// path) lifted verbatim from llama.cpp quants.c:382, raw byte offsets.
// block_q5_1 = 24B: d@0(fp16) m@2(fp16) qh[4]@4 qs[16]@8.
// block_q8_1 = 36B: d@0(fp16) s@2(fp16) qs[32]@4.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 32; const int Q5 = 24, Q8 = 36;
  const int nb = (int)n / qk; float sumf = 0; size_t vl;
  size_t vlenb = __riscv_vlenb();
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q5, *yb = vy + (size_t)ib * Q8;
    const uint8_t *xqs = xb + 8, *xqh = xb + 4;
    const int8_t *yqs = (const int8_t *)(yb + 4);
    uint16_t xdh, xmh, ydh, ysh;
    memcpy(&xdh, xb + 0, 2); memcpy(&xmh, xb + 2, 2);
    memcpy(&ydh, yb + 0, 2); memcpy(&ysh, yb + 2, 2);
    vl = qk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
    vint8m2_t v0c;
    if (vlenb == 16) { v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h); }
    else { v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32); v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l); }
    vl = qk;
    vbool4_t qh = __riscv_vlm_v_b4(xqh, vl);
    vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);
    _Float16 xdf, xmf, ydf, ysf;
    memcpy(&xdf, &xdh, 2); memcpy(&xmf, &xmh, 2);
    memcpy(&ydf, &ydh, 2); memcpy(&ysf, &ysh, 2);
    sumf += ((float)xdf * (float)ydf) * sumi + (float)xmf * (float)ysf;
  }
  *s = sumf;
}
