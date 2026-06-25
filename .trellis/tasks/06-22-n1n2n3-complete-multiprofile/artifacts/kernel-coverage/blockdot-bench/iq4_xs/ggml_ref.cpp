// ggml's SHIPPED RVV iq4_xs_q8_K kernel, _vl128 variant, lifted VERBATIM from
// llama.cpp ggml-cpu/arch/riscv/quants.c:5608, re-expressed against raw byte offsets.
// block_iq4_xs=136B: d@0(fp16) scales_h@2(u16) scales_l[4]@4(u8) qs[128]@8(u8).
// block_q8_K=292B: d@0(float32) qs[256]@4(int8) bsums[16]@260(int16). iq4_xs IGNORES
// bsums. FP4 16-entry codebook (kvalues_iq4nl) gather. NOTE the FOLD ASSOCIATION:
// ggml integer-accumulates sumi1/sumi2 across all 8 sub-blocks then does ONE fp
// mul-add per super-block (sumf += d * (sumi1+sumi2)). OURS distributes the fold
// per-sub-block (sumf += d1*(float)sumi). Same value in exact arithmetic, different
// fp32 rounding -> expect a small NONZERO rel-norm vs THIS verbatim ref (the tq1_0
// situation); the matched-association ref (ggml_ref_matched.cpp) collapses it to 0.0.
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define QK_K 256

// kvalues_iq4nl: the 16-entry non-linear FP4 codebook (shared by iq4_nl + iq4_xs).
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int WB = 136, YB = 292;
  const int nb = (int)n / QK_K;

  const vint8m4_t values = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
  float sumf = 0;

  for (int ibl = 0; ibl < nb; ++ibl) {
    const uint8_t *xb = vx + (size_t)ibl * WB, *yb = vy + (size_t)ibl * YB;
    const int8_t  *q8  = (const int8_t *)(yb + 4);      // q8_K.qs[256]@4
    const uint8_t *iq4 = xb + 8;                        // iq4_xs.qs[128]@8
    uint16_t h;        memcpy(&h, xb + 2, 2);           // scales_h u16 @2
    const uint8_t *scales_l = xb + 4;                   // scales_l[4] @4

    int sumi1 = 0, sumi2 = 0;
    for (int ib = 0; ib < QK_K / 64; ++ib) {            // 4 iters, 2 sub-blocks each
      // Load the packed weights (32 bytes = 64 nibbles).
      const vuint8m2_t iq4_packed = __riscv_vle8_v_u8m2(iq4, 32);
      iq4 += 32;

      // Unpack the weight blocks.
      const vuint8m2_t iq4bits_lo = __riscv_vand_vx_u8m2(iq4_packed, 0xf, 32);
      const vuint8m2_t iq4bits_hi = __riscv_vsrl_vx_u8m2(iq4_packed, 4, 32);
      const vuint8m4_t iq4bits = __riscv_vcreate_v_u8m2_u8m4(iq4bits_lo, iq4bits_hi);
      const vuint8m4_t iq4bits_reorder = __riscv_vcreate_v_u8m1_u8m4(
          __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 0), 16),
          __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 2), 16),
          __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 1), 16),
          __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 3), 16)
      );
      const vint8m4_t iq4b = __riscv_vrgather_vv_i8m4(values, iq4bits_reorder, 64);

      // Multiply with activations.
      const vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 64);
      q8 += 64;
      const vint16m8_t prod = __riscv_vwmul_vv_i16m8(iq4b, q8b, 64);

      // Reduce separately.
      const int acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));

      const int ls1 = ((scales_l[ib] & 0xf)  | ((h << 4) & 0x30)) - 32;
      const int ls2 = ((scales_l[ib] >>  4)  | ((h << 2) & 0x30)) - 32;
      h >>= 4;

      sumi1 += acc0 * ls1;
      sumi2 += acc1 * ls2;

      __asm__ __volatile__("" ::: "memory");
    }

    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 0, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    sumf += (float)xd * yd * (sumi1 + sumi2);
  }

  *s = sumf;
}
