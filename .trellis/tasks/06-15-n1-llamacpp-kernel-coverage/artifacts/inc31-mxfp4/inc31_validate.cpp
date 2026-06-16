// INC-31 mxfp4 x q8_0 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.mxfp4_q8_0_block_dot op
// (the COMPLETE ggml ggml_vec_dot_mxfp4_q8_0 block kernel: AoS QK=32 block loop +
// per-block E8M0 weight scale + q8_0 fp16 scale + FP4 codebook decode + fp32
// accumulation + *s store) computes the SAME fp32 result `*s` as ggml's OWN mxfp4
// RVV kernel AND its _generic reference, BIT-FOR-BIT (memcmp of the float bits),
// over random block_mxfp4 x block_q8_0 arrays at n in {32,64,...,8192} + named edge
// cases (all 16 FP4 codebook entries exercised, the full E8M0 exponent range incl.
// e=0/min/max, q8 +/-127). THREE negative controls must FAIL -> proves the FP4
// codebook AND the E8M0 half-scale convention are the live, load-bearing mechanisms.
//
// mxfp4 opens the LAST structural quantization class: FP4 (e2m1) weights with an
// E8M0 shared-exponent block scale. block_mxfp4 = { uint8_t e; uint8_t qs[16] }
// (sizeof 17): a SINGLE 8-bit E8M0 exponent at +0, then 16 packed FP4 nibble bytes
// at +1 (NOT iq4_nl's {fp16 d; ...} 18-byte block). The 4-bit nibble INDEXES a
// 16-entry NON-LINEAR int8 CODEBOOK kvalues_mxfp4[16] = 2*E2M1 (a LOOKUP, REUSING
// the iq4_nl gather mechanism). The block scale is GGML_E8M0_TO_FP32_HALF(e) =
// 2^(e-128) (the HALF form matching the DOUBLED codebook). The fold is mxfp4's
// scales-first order: sumf += d * (sumi1 + sumi2), where d = d_y * scale_x.
// (ggml _generic quants.c:247-276; ggml RVV vl128 arch/riscv/quants.c:6470-6523.)
//
// REFERENCE: ggml's _generic AND a faithful copy of ggml's RVV mxfp4 method (vand/
// vsrl nibble split -> vrgather through the kvalues_mxfp4 table -> vwmul/vwmacc ->
// vwredsum), both with an order-independent integer sumi and the IDENTICAL fp32
// statement, so all three are the same fp32 result bit-for-bit. FP16->FP32 mirrors
// the board's scalar _Float16 widening (lossless). The E8M0->fp32 half mirrors
// ggml's EXACT bit construction. Whole TU built with the same -ffp-contract setting.
//
// The kernel under test is the UNMODIFIED, compiler-emitted mxfp4 kernel
// (mxfp4_m1_elided.cpp) -- every line tagged source_op=tcrv_rvv.mxfp4_q8_0_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_mxfp4 = { uint8_t e; uint8_t qs[16]; }  sizeof 17, nibbles @1
//   block_q8_0  = { fp16 d;    int8_t  qs[32]; }  sizeof 34, quants @2
static const int QK = 32;
static const int MXFP4_STRIDE = 17;
static const int Q8_STRIDE = 34;

// ---- ggml's 16-entry FP4 (e2m1) codebook (ggml-common.h:1116-1118) -----------
// kvalues_mxfp4 = 2 * E2M1_float = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}
static const int8_t kvalues_mxfp4[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};
// The WRONG codebook for negative control 1: the LINEAR nibble-8 decode (what
// q4_0 uses). If the codebook were NOT load-bearing, swapping it would not change
// the result -- it MUST.
static const int8_t kvalues_linear[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7};

// ---- ggml's E8M0 -> fp32 reconstruction (ggml-impl.h) ------------------------
// HALF form (the REAL mxfp4 scale): 2^(e-128), matching the DOUBLED codebook.
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
  uint32_t bits;
  if (x < 2)
    bits = 0x00200000u << x; // 2^-128 / 2^-127 denormal patterns
  else
    bits = (uint32_t)(x - 1) << 23; // 2^(e-128) normalized
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}
// FULL form (the WRONG scale for negative control 3): 2^(e-127). This is what the
// task prose loosely says; the REAL kernel uses the half form. A kernel built with
// the full scale would DOUBLE every result -> it MUST diverge.
static inline float ggml_e8m0_to_fp32_full(uint8_t x) {
  uint32_t bits;
  if (x == 0)
    bits = 0x00400000; // 2^-126 denormal
  else
    bits = (uint32_t)x << 23; // 2^(e-127) normalized
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
extern "C" void
tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's REAL mxfp4 RVV method (arch/riscv/quants.c vl128, half-block) -----
// Parameterised by the codebook table + the E8M0 scale fn so the negative controls
// can swap each independently.
typedef float (*e8m0_fn)(uint8_t);
static void ggml_rvv_tab(int n, float *s, const void *vx, const void *vy,
                         const int8_t *table, e8m0_fn e8m0) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  vint8m1_t values = __riscv_vle8_v_i8m1(table, 16);
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * MXFP4_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    uint8_t e = xb[0];                          // E8M0 exponent @0
    const uint8_t *xqs = xb + 1;                // packed FP4 nibbles @1
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t ydh;
    std::memcpy(&ydh, yb + 0, 2);

    size_t vl = QK / 2; // 16
    vuint8m1_t packed = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yqs, vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1(yqs + 16, vl);
    vuint8m1_t iL = __riscv_vand_vx_u8m1(packed, 0x0F, vl);
    vuint8m1_t iH = __riscv_vsrl_vx_u8m1(packed, 4, vl);
    vint8m1_t v0 = __riscv_vrgather_vv_i8m1(values, iL, vl);
    vint8m1_t v1 = __riscv_vrgather_vv_i8m1(values, iH, vl);
    vint16m2_t mul = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    mul = __riscv_vwmacc_vv_i16m2(mul, v1, y1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

    sumf += (float)sumi * (e8m0(e) * fp16_to_fp32(ydh));
  }
  *s = sumf;
}

static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  ggml_rvv_tab(n, s, vx, vy, kvalues_mxfp4, ggml_e8m0_to_fp32_half);
}

// ---- ggml's _generic (quants.c:247-276, the byte-exact same math) ------------
// d = d_y * scale_x; sumf += d * (sumi1 + sumi2).
static void ggml_generic_tab(int n, float *s, const void *vx, const void *vy,
                             const int8_t *table, e8m0_fn e8m0) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * MXFP4_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    uint8_t e = xb[0];
    const uint8_t *xqs = xb + 1;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t ydh;
    std::memcpy(&ydh, yb + 0, 2);
    const float d = fp16_to_fp32(ydh) * e8m0(e);
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < QK / 2; ++j) {
      sumi1 += yqs[j + 0] * table[xqs[j] & 0xf];
      sumi2 += yqs[j + QK / 2] * table[xqs[j] >> 4];
    }
    sumf += d * (sumi1 + sumi2);
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_tab(n, s, vx, vy, kvalues_mxfp4, ggml_e8m0_to_fp32_half);
}

// ---- random block_mxfp4 / block_q8_0 array generators ------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

// The E8M0 exponent is biased; a model's scales cluster near 1.0 (e ~ 127). To
// keep the products fp32-representable but EXERCISE a wide exponent range, draw e
// from a band around the bias 127 (+/- ~24). Edge fillers pin the extremes (0/255).
static inline uint8_t rand_e8m0() {
  return (uint8_t)(103 + (xrand() % 49)); // [103, 151] -> 2^[-25, 23]
}

static void fill_mxfp4(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * MXFP4_STRIDE;
    b[0] = rand_e8m0();                     // E8M0 exponent
    for (int j = 0; j < QK / 2; j++)
      b[1 + j] = (uint8_t)(xrand() % 256);  // packed FP4 nibbles (every value)
  }
}

static void fill_q8_0(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q8_STRIDE;
    put_fp16(b + 0); // d
    for (int j = 0; j < QK; j++)
      b[2 + j] = (uint8_t)(int8_t)(xrand() % 256);
  }
}

// Edge-case fillers.
//   xmode 0: all nibbles 0x00   -> both halves index entry 0  (kvalues[0] = 0).
//   xmode 1: all nibbles 0xFF   -> both halves index entry 15 (kvalues[15] = -12).
//   xmode 2: byte j = (j&0xF)|((j&0xF)<<4) marching the nibble through 0..15 ->
//            EXERCISES ALL 16 FP4 CODEBOOK ENTRIES (the load-bearing gather coverage).
//   emode is the E8M0 exponent: 0 -> e=0 (min, denormal 2^-128); 1 -> e=1
//   (2^-127); 254 -> e=254 (large normal); 255 -> e=255 (max); 127 -> e=127 (~1.0).
static void fill_mxfp4_edge(uint8_t *buf, int nb, int xmode, uint8_t e) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * MXFP4_STRIDE;
    b[0] = e;
    for (int j = 0; j < QK / 2; j++) {
      uint8_t v;
      if (xmode == 0) v = 0x00;
      else if (xmode == 1) v = 0xFF;
      else { uint8_t nib = (uint8_t)(j & 0xF); v = (uint8_t)(nib | (nib << 4)); }
      b[1 + j] = v;
    }
  }
}

static void fill_q8_0_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q8_STRIDE;
    put_fp16(b + 0);
    for (int j = 0; j < QK; j++)
      b[2 + j] = (uint8_t)(int8_t)((mode == 0) ? 127 : ((j & 1) ? -128 : 127));
  }
}

#ifndef CONTRACT_REFS
#define CONTRACT_REFS 1
#endif

static int check_buffers(int n, uint8_t *vx, uint8_t *vy, const char *tag,
                         int *generic_delta) {
  float so = 0, sg = 0, sr = 0;
  our_kernel(n, &so, vx, vy);
  ggml_rvv(n, &sg, vx, vy);
  ggml_generic(n, &sr, vx, vy);
  int fail = 0;
  if (std::memcmp(&so, &sg, 4) != 0) {
    printf("  %-26s n=%-6d FAIL vs ggml-rvv : ours=%.9g ggml=%.9g\n", tag, n, so, sg);
    fail = 1;
  }
  if (std::memcmp(&so, &sr, 4) != 0) {
    *generic_delta += 1;
    if (CONTRACT_REFS) {
      printf("  %-26s n=%-6d FAIL vs _generic : ours=%.9g gen=%.9g\n", tag, n, so, sr);
      fail = 1;
    }
  }
  return fail;
}

static int check_random(int n, int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * MXFP4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_mxfp4(vx, nb);
  fill_q8_0(vy, nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int xmode, uint8_t e, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * MXFP4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_mxfp4_edge(vx, nb, xmode, e);
  fill_q8_0_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-31 mxfp4 byte-exact (PRIMARY target: ggml's REAL mxfp4 RVV "
         "method) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }
  // Edge cases: the marching-nibble pattern that exercises ALL 16 FP4 codebook
  // entries + the all-0 / all-15 nibbles, crossed with the FULL E8M0 exponent range
  // (e=0 denormal min, e=1, e=127 ~1.0, e=254, e=255 max) + q8 saturation. The
  // marching x e-extreme x q8 saturation cases are the strongest combined probe.
  uint8_t e_set[] = {0, 1, 127, 254, 255};
  for (int n : n_set) {
    for (uint8_t e : e_set) {
      total += 4;
      char tag[64];
      snprintf(tag, sizeof tag, "nib=0 e=%u q8=+127", e);
      fails += check_edge(n, 0, e, 0, tag, &generic_delta);
      snprintf(tag, sizeof tag, "nib=15 e=%u q8=+/-", e);
      fails += check_edge(n, 1, e, 1, tag, &generic_delta);
      snprintf(tag, sizeof tag, "all16 e=%u q8=+/-", e);
      fails += check_edge(n, 2, e, 1, tag, &generic_delta);
      snprintf(tag, sizeof tag, "all16 e=%u q8=+127", e);
      fails += check_edge(n, 2, e, 0, tag, &generic_delta);
    }
  }
  printf("checked %d cases, %d failures (vs ggml's REAL mxfp4 RVV method)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING FP4 codebook proof): keep the EXACT SAME
  // input bytes, but compute the reference with the WRONG (linear nibble-8) codebook.
  // Our kernel embeds the real kvalues_mxfp4 table, so it MUST diverge -> the
  // codebook is the live mechanism, not dead.
  {
    int n = 2048, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * MXFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    // Random nibbles + q8 all +127 so the integer sum is NON-ZERO (the symmetric
    // codebook makes the marching/symmetric pattern sum to 0, which would make any
    // scale/codebook control vacuous). Pin e=127 (~1.0) so the products are tame.
    fill_mxfp4(vx, nb);
    for (int i = 0; i < nb; i++) vx[i * MXFP4_STRIDE] = 127;
    fill_q8_0_edge(vy, nb, 0);       // q8 +127
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_tab(n, &s_real, vx, vy, kvalues_mxfp4, ggml_e8m0_to_fp32_half);
    ggml_generic_tab(n, &s_wrong, vx, vy, kvalues_linear, ggml_e8m0_to_fp32_half);
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-1 setup FAILED: ours != real-codebook ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 1 FAILED: WRONG (linear) codebook still matched "
             "(vacuous -- the FP4 codebook is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 1 OK: WRONG (linear nibble-8) codebook diverges "
             "(ours=%.9g real=%.9g linear=%.9g) -> the FP4 codebook is load-bearing\n",
             so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the LOAD-BEARING E8M0 HALF-scale proof): keep the EXACT
  // SAME input bytes, but compute the reference with the WRONG (FULL 2^(e-127))
  // E8M0 scale instead of the HALF (2^(e-128)). Our kernel embeds the half form, so
  // it MUST diverge from a full-scale reference -> the half convention (the crux of
  // mxfp4 vs the task prose's loose 2^(e-127)) is the live, load-bearing choice.
  {
    int n = 2048, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * MXFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    // Random nibbles + q8 +127 (NON-ZERO integer sum) at e=130 (normal, where
    // half=2^2=4 differs from full=2^3=8 by exactly 2x) so the half/full scales
    // genuinely diverge.
    fill_mxfp4(vx, nb);
    for (int i = 0; i < nb; i++) vx[i * MXFP4_STRIDE] = 130;
    fill_q8_0_edge(vy, nb, 0);
    float so = 0, s_half = 0, s_full = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_tab(n, &s_half, vx, vy, kvalues_mxfp4, ggml_e8m0_to_fp32_half);
    ggml_generic_tab(n, &s_full, vx, vy, kvalues_mxfp4, ggml_e8m0_to_fp32_full);
    if (std::memcmp(&so, &s_half, 4) != 0) {
      printf("NEG-CTRL-2 setup FAILED: ours != half-scale ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_full, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: WRONG (full 2^(e-127)) E8M0 scale still "
             "matched (vacuous -- the half scale is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: WRONG (full 2^(e-127)) E8M0 scale diverges "
             "(ours=%.9g half=%.9g full=%.9g) -> the E8M0 HALF scale is load-bearing\n",
             so, s_half, s_full);
    }
    free(vx);
    free(vy);
  }

  // Negative control 3: flip ONE nibble of vx; our kernel must DIFFER from a fresh
  // ggml run on the UNperturbed data (proves the check is non-vacuous: the FP4
  // nibbles are actually consumed).
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * MXFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_mxfp4(vx, nb);
    fill_q8_0_edge(vy, nb, 0); // q8 = +127 so the perturbed product is non-zero
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[1] ^= 0x07; // flip part of the low nibble of element 0 in block 0 (entry @+1)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL 3 FAILED: perturbed nibble still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 3 OK: flipped nibble diverges (the FP4 nibbles are "
             "consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL mxfp4 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("mxfp4 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
