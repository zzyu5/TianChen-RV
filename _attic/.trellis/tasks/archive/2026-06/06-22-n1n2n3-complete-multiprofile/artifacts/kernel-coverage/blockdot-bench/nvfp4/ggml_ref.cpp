// nvfp4_q8_0 correctness ORACLE = ggml's SHIPPED riscv kernel, which is the GENERIC
// (scalar) path: ggml ships NO RVV vec_dot for nvfp4 on riscv (arch-fallback.h maps
// ggml_vec_dot_nvfp4_q8_0_generic -> ggml_vec_dot_nvfp4_q8_0; only arm has a vector
// kernel). So generic IS the real board kernel here. Lifted verbatim from
// llama.cpp ggml-cpu/quants.c:279. QK_NVFP4=64 superblock, QK_NVFP4_SUB=16,
// block_nvfp4=36B {d[4] UE4M3@0..3; qs[32]@4}, two block_q8_0(34B) per superblock.
// HONESTY: this is a SCALAR baseline -> the ours/ggml ratio is vector-vs-scalar, NOT
// the vector-vs-vector gather-shape comparison that produced the iq4_nl/mxfp4 WINs.
#include <stdint.h>
#include <string.h>
#include <math.h>

static const int8_t kvalues_mxfp4[16] = {
  0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ggml_ue4m3_to_fp32 (ggml-impl.h)
static inline float ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  int exp = (x >> 3) & 0xF;
  int man = x & 0x7;
  float raw;
  if (exp == 0) raw = ldexpf((float)man, -9);
  else          raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  return raw * 0.5f;
}

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
#define WB 36
#define YB 34

extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / QK_NVFP4;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * WB;        // block_nvfp4
    const uint8_t *d4 = xb + 0;                       // UE4M3 scales d[4]
    const uint8_t *qs = xb + 4;                       // packed FP4 nibbles qs[32]
    for (int s_idx = 0; s_idx < 4; ++s_idx) {
      const float d = ue4m3_to_fp32(d4[s_idx]);
      const int q8_block = s_idx / 2;
      const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
      const uint8_t *yb = vy + (size_t)(2 * ib + q8_block) * YB;
      _Float16 yd16; uint16_t t; memcpy(&t, yb + 0, 2); memcpy(&yd16, &t, 2);
      const float dy = (float)yd16;
      const int8_t *yq = (const int8_t *)(yb + 2);
      int sumi_lo = 0, sumi_hi = 0;
      for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
        const uint8_t qv = qs[s_idx * (QK_NVFP4_SUB / 2) + j];
        sumi_lo += yq[q8_off + j + 0]                * kvalues_mxfp4[qv & 0xf];
        sumi_hi += yq[q8_off + j + QK_NVFP4_SUB / 2] * kvalues_mxfp4[qv >> 4];
      }
      sumf += dy * d * (sumi_lo + sumi_hi);
    }
  }
  *s = sumf;
}
