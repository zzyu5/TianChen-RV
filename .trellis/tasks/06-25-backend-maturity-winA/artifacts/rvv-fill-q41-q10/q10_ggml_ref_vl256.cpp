// ggml's SHIPPED RVV q1_0_q8_0 _vl256 body — the path ggml's runtime dispatch
// (ggml_vec_dot_q1_0_q8_0: switch on __riscv_vlenb()*8, case 256 -> _vl256) selects
// on a VLEN256 board (k1). Lifted verbatim from llama.cpp
// ggml/src/ggml-cpu/arch/riscv/quants.c:484, re-expressed against raw byte offsets
// (block_q1_0=18B: d@0 qs[16]@2 ; block_q8_0=34B: d@0 qs[32]@2). QK1_0=128, one
// super-block spans 4 q8_0 blocks. e8m1 VLMAX=32 at VLEN256 (FULL m1 register), so the
// 32-lane sub-block fills one whole m1 strip — the SAME shape OUR m1 emit produces.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

extern "C" void kern_ggml_vl256(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 128;
  const int WB = 18, YB = 34;
  const int nb = (int)n / qk;
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  const vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, 1);
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * WB;
    uint16_t xdh; memcpy(&xdh, xb + 0, 2);
    _Float16 xf; memcpy(&xf, &xdh, 2);
    const float d0 = (float)xf;
    float acc = 0;
    for (int k = 0; k < 4; ++k) {
      const uint8_t *yb = vy + ((size_t)ib * 4 + k) * YB;
      const uint8_t *qsbits = xb + 2 + 4 * k;
      const int8_t  *yqs   = (const int8_t *)(yb + 2);
      uint16_t ydh; memcpy(&ydh, yb + 0, 2);
      _Float16 yf; memcpy(&yf, &ydh, 2);
      const vbool8_t is_not_zero = __riscv_vlm_v_b8(qsbits, vl32);
      const vint8m1_t qy = __riscv_vle8_v_i8m1(yqs, vl32);
      const vint8m1_t neg_qy = __riscv_vneg_v_i8m1(qy, vl32);
      const vint8m1_t sy = __riscv_vmerge_vvm_i8m1(neg_qy, qy, is_not_zero, vl32);
      const vint16m1_t red = __riscv_vwredsum_vs_i8m1_i16m1(sy, zero, vl32);
      acc += (float)yf * (float)__riscv_vmv_x_s_i16m1_i16(red);
    }
    sumf += d0 * acc;
  }
  *s = sumf;
}
