// ggml's SHIPPED RVV iq4_xs_q8_K kernel, _vl256 variant (k1/VLEN256 dispatch path).
// Lifted verbatim from llama.cpp quants.c:5674, raw byte offsets.
// block_iq4_xs=136B: d@0(fp16) scales_h@2(u16) scales_l[4]@4(u8) qs[128]@8(u8).
// block_q8_K=292B: d@0(float32) qs[256]@4(int8) bsums[16]@260(int16). iq4_xs IGNORES bsums.
// _vl256: ONE i8m4 128-nibble gather (QK_K/128=2 iters), vrgatherei16 u64m4 reorder,
//   4-way vget/vwredsum, scales_l[ib*2+0/1], h>>=8.
// FOLD: ggml integer-accumulates sumi1..4 across both iters then ONE fp mul-add per
//   super-block; OURS distributes per-sub-block -> small nonzero rel-norm vs verbatim
//   (fp reassoc), 0.0 vs matched-assoc ref.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define QK_K 256

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int WB = 136, YB = 292;
  const int nb = (int)n / QK_K;

  const vint8m4_t values = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
  float sumf = 0;

  uint16_t index[16] = { 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
  vuint16m1_t i_vec = __riscv_vle16_v_u16m1(index, 16);

  for (int ibl = 0; ibl < nb; ++ibl) {
    const uint8_t *xb = vx + (size_t)ibl * WB, *yb = vy + (size_t)ibl * YB;
    const int8_t  *q8  = (const int8_t *)(yb + 4);
    const uint8_t *iq4 = xb + 8;
    uint16_t h;        memcpy(&h, xb + 2, 2);
    const uint8_t *scales_l = xb + 4;

    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    for (int ib = 0; ib < QK_K / 128; ++ib) {
      vuint8m2_t iq4_packed = __riscv_vle8_v_u8m2(iq4, 64);
      iq4 += 64;

      vuint8m2_t iq4bits_lo = __riscv_vand_vx_u8m2(iq4_packed, 0xf, 64);
      vuint8m2_t iq4bits_hi = __riscv_vsrl_vx_u8m2(iq4_packed, 4, 64);
      vuint8m4_t iq4bits = __riscv_vcreate_v_u8m2_u8m4(iq4bits_lo, iq4bits_hi);
      vuint8m4_t iq4bits_reorder = __riscv_vreinterpret_v_u64m4_u8m4(
          __riscv_vrgatherei16_vv_u64m4(__riscv_vreinterpret_v_u8m4_u64m4(iq4bits), i_vec, 16));
      vint8m4_t iq4b = __riscv_vrgather_vv_i8m4(values, iq4bits_reorder, 128);

      __asm__ __volatile__("" ::: "memory");

      vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 128);
      vint16m8_t prod = __riscv_vwmul_vv_i16m8(iq4b, q8b, 128);
      q8 += 128;

      __asm__ __volatile__("" ::: "memory");

      int acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int acc2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int acc3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));

      int ls1 = ((scales_l[ib * 2 + 0] & 0xf)  | ((h << 4) & 0x30)) - 32;
      int ls2 = ((scales_l[ib * 2 + 0] >>  4)  | ((h << 2) & 0x30)) - 32;
      int ls3 = ((scales_l[ib * 2 + 1] &  0xf) | ((h << 0) & 0x30)) - 32;
      int ls4 = ((scales_l[ib * 2 + 1] >>  4)  | ((h >> 2) & 0x30)) - 32;
      h >>= 8;

      sumi1 += acc0 * ls1;
      sumi2 += acc1 * ls2;
      sumi3 += acc2 * ls3;
      sumi4 += acc3 * ls4;

      __asm__ __volatile__("" ::: "memory");
    }

    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 0, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    sumf += (float)xd * yd * (sumi1 + sumi2 + sumi3 + sumi4);
  }

  *s = sumf;
}
