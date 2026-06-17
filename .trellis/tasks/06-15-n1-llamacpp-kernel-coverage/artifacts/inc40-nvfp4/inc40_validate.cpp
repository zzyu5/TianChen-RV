// INC-40 nvfp4 x q8_0 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.nvfp4_q8_0_block_dot op
// (the COMPLETE ggml ggml_vec_dot_nvfp4_q8_0 block kernel: super-block QK=64 loop +
// four 16-element sub-blocks + per-sub-block UE4M3 weight scale + q8_0 fp16 scale +
// FP4 codebook decode + fp32 accumulation + *s store) computes the SAME fp32 result
// `*s` as ggml's nvfp4 kernel BIT-FOR-BIT (memcmp of the float bits). There is NO
// arch/riscv nvfp4 kernel -- arch-fallback.h aliases ggml_vec_dot_nvfp4_q8_0 to
// ggml_vec_dot_nvfp4_q8_0_generic, so _generic IS the real board kernel AND the
// oracle. We compare against a faithful transcription of _generic AND an independent
// RVV transcription (the same vand/vsrl/vrgather/vwmul/vwmacc/vwredsum core ggml's
// other codebook kernels use), both bit-for-bit, over random + edge cases.
//
// nvfp4 is NVIDIA's FP4 -- the SECOND FP4-class kernel. block_nvfp4 =
// { uint8_t d[4]; uint8_t qs[32] } (QK_NVFP4=64, sizeof 36): FOUR UE4M3 sub-block
// scales at +0..3, then 32 packed FP4 nibble bytes at +4 (8 per 16-element
// sub-block). The 4-bit nibble INDEXES the SAME 16-entry kvalues_mxfp4[16] = 2*E2M1
// codebook nvfp4 REUSES from mxfp4. ONE super-block (64 elems) spans TWO block_q8_0
// blocks: sub-block s reads q8 block (2*ib + s/2) at half-offset (s%2)*16. The block
// scale is ggml_ue4m3_to_fp32(e) (UNSIGNED 4-exp/3-man fp8, ldexpf decode, *0.5f).
// The fold is ggml's order: sumf += dy * d * (sumi_lo + sumi_hi).
// (ggml _generic quants.c:278-312; ggml-impl.h ggml_ue4m3_to_fp32; ggml-common.h.)
//
// CRITICAL (the buffer-sizing landmine): nb = n/64 super-blocks, but the q8 stream
// is n/32 = 2*nb q8_0 blocks (sub-blocks s=2,3 read q8 block 2*ib+1). Under-sizing
// the q8 buffer to nb would make OUR kernel AND the reference read the same OOB
// addresses -> a FALSE pass that never validates the s=2,3 path. We size 2*nb.
//
// The kernel under test is the UNMODIFIED, compiler-emitted nvfp4 kernel
// (nvfp4_kernel.cpp, #include'd below for libm/<cmath> in scope) -- every line tagged
// source_op=tcrv_rvv.nvfp4_q8_0_block_dot. ldexpf is a structured emitc.call_opaque
// to an external function (the SAME category as the sanctioned fp16 read), not a
// raw() escape -- two sanctioned opaque scalar pieces (the q8 fp16 read + ldexpf).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// The compiler-emitted kernel, textually included so its `ldexpf` resolves against
// <cmath> in this TU (libm linked via -lm). It is byte-identical to a fresh
// tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate regen.
#include "nvfp4_kernel.cpp"

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_nvfp4 = { uint8_t d[4]; uint8_t qs[32]; }  sizeof 36, nibbles @4
//   block_q8_0  = { fp16 d;       int8_t  qs[32]; }  sizeof 34, quants @2
static const int QK = 64;        // QK_NVFP4
static const int QK_SUB = 16;    // QK_NVFP4_SUB
static const int NVFP4_STRIDE = 36;
static const int Q8_STRIDE = 34;
static const int N_SUB = QK / QK_SUB;        // 4 sub-blocks per super-block
static const int SUB_HALF = QK_SUB / 2;      // 8

// ---- ggml's 16-entry FP4 (e2m1) codebook (ggml-common.h:1116-1118) -----------
// nvfp4 REUSES kvalues_mxfp4 = 2*E2M1 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}.
static const int8_t kvalues_mxfp4[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};
// The WRONG codebook for negative control 1: the LINEAR nibble-8 decode (what q4_0
// uses). If the codebook were NOT load-bearing, swapping it would not change the
// result -- it MUST.
static const int8_t kvalues_linear[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7};

// ---- ggml's UE4M3 -> fp32 reconstruction (ggml-impl.h:502-515) ----------------
// HALF form (the REAL nvfp4 scale): raw * 0.5f, matching the DOUBLED codebook.
static inline float ggml_ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  int   exp = (x >> 3) & 0xF;
  int   man = x & 0x7;
  float raw;
  if (exp == 0) {
    raw = ldexpf((float) man, -9);
  } else {
    raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
  }
  return raw * 0.5f;
}
// FULL form (the WRONG scale for negative control 2): drop the *0.5f. A kernel built
// without the half compensation DOUBLES every result -> it MUST diverge.
static inline float ggml_ue4m3_to_fp32_full(uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  int   exp = (x >> 3) & 0xF;
  int   man = x & 0x7;
  float raw;
  if (exp == 0) {
    raw = ldexpf((float) man, -9);
  } else {
    raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
  }
  return raw; // NO *0.5f
}

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

typedef float (*ue4m3_fn)(uint8_t);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's _generic (quants.c:278-312, the byte-exact same math) ------------
// Parameterised by codebook + UE4M3 scale fn so the negative controls swap each.
static void ggml_generic_tab(int n, float *s, const void *vx, const void *vy,
                             const int8_t *table, ue4m3_fn ue4m3) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + (size_t)ib * NVFP4_STRIDE;
    for (int s_idx = 0; s_idx < N_SUB; ++s_idx) {
      const float d = ue4m3(xb[s_idx]);
      const int q8_block = s_idx / 2;
      const int q8_off   = (s_idx % 2) * QK_SUB;
      const uint8_t *yb = y + (size_t)(2 * ib + q8_block) * Q8_STRIDE;
      uint16_t ydh;
      std::memcpy(&ydh, yb + 0, 2);
      const float dy = fp16_to_fp32(ydh);
      const uint8_t *xqs = xb + 4 + s_idx * (QK_SUB / 2);
      const int8_t *yqs = (const int8_t *)(yb + 2);
      int sumi_lo = 0, sumi_hi = 0;
      for (int j = 0; j < QK_SUB / 2; ++j) {
        const uint8_t qv = xqs[j];
        sumi_lo += yqs[q8_off + j +               0] * table[qv & 0xf];
        sumi_hi += yqs[q8_off + j + QK_SUB / 2]      * table[qv >>  4];
      }
      sumf += dy * d * (sumi_lo + sumi_hi);
    }
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_tab(n, s, vx, vy, kvalues_mxfp4, ggml_ue4m3_to_fp32);
}

// ---- an independent RVV transcription (same codebook gather core) -------------
static void ggml_rvv_tab(int n, float *s, const void *vx, const void *vy,
                         const int8_t *table, ue4m3_fn ue4m3) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  vint8m1_t values = __riscv_vle8_v_i8m1(table, 16);
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + (size_t)ib * NVFP4_STRIDE;
    for (int s_idx = 0; s_idx < N_SUB; ++s_idx) {
      const float d = ue4m3(xb[s_idx]);
      const int q8_block = s_idx / 2;
      const int q8_off   = (s_idx % 2) * QK_SUB;
      const uint8_t *yb = y + (size_t)(2 * ib + q8_block) * Q8_STRIDE;
      uint16_t ydh;
      std::memcpy(&ydh, yb + 0, 2);
      const float dy = fp16_to_fp32(ydh);
      const uint8_t *xqs = xb + 4 + s_idx * (QK_SUB / 2);
      const int8_t *yqs = (const int8_t *)(yb + 2 + q8_off);

      size_t vl = SUB_HALF; // 8
      vuint8m1_t packed = __riscv_vle8_v_u8m1(xqs, vl);
      vint8m1_t y0 = __riscv_vle8_v_i8m1(yqs, vl);
      vint8m1_t y1 = __riscv_vle8_v_i8m1(yqs + SUB_HALF, vl);
      vuint8m1_t iL = __riscv_vand_vx_u8m1(packed, 0x0F, vl);
      vuint8m1_t iH = __riscv_vsrl_vx_u8m1(packed, 4, vl);
      vint8m1_t v0 = __riscv_vrgather_vv_i8m1(values, iL, vl);
      vint8m1_t v1 = __riscv_vrgather_vv_i8m1(values, iH, vl);
      vint16m2_t mul = __riscv_vwmul_vv_i16m2(v0, y0, vl);
      mul = __riscv_vwmacc_vv_i16m2(mul, v1, y1, vl);
      vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(mul, zero, vl);
      int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

      sumf += dy * d * (float)sumi;
    }
  }
  *s = sumf;
}

static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  ggml_rvv_tab(n, s, vx, vy, kvalues_mxfp4, ggml_ue4m3_to_fp32);
}

// ---- random block_nvfp4 / block_q8_0 array generators ------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

// A UE4M3 scale code drawn to cluster near 1.0 (bias 7 -> e ~ 0x38) but spanning a
// wide range; edge fillers pin the extremes (0x00/0x7F/denormals).
static inline uint8_t rand_ue4m3() {
  return (uint8_t)(0x20 + (xrand() % 0x50)); // [0x20,0x6F] -> ~[2^-3, 224]
}

static void fill_nvfp4(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * NVFP4_STRIDE;
    for (int s = 0; s < N_SUB; s++)
      b[s] = rand_ue4m3();                       // four UE4M3 sub-block scales
    for (int j = 0; j < QK / 2; j++)
      b[4 + j] = (uint8_t)(xrand() % 256);       // packed FP4 nibbles (every value)
  }
}

// nb = super-blocks; the q8 stream is q8nb = 2*nb q8_0 blocks (the buffer-sizing fix).
static void fill_q8_0(uint8_t *buf, int q8nb) {
  for (int i = 0; i < q8nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8_STRIDE;
    put_fp16(b + 0); // d
    for (int j = 0; j < 32; j++)
      b[2 + j] = (uint8_t)(int8_t)(xrand() % 256);
  }
}

// Edge-case fillers.
//   xmode 0: all nibbles 0x00   -> both halves index entry 0  (kvalues[0] = 0).
//   xmode 1: all nibbles 0xFF   -> both halves index entry 15 (kvalues[15] = -12).
//   xmode 2: byte j = (j&0xF)|((j&0xF)<<4) marching the nibble through 0..15 ->
//            EXERCISES ALL 16 FP4 CODEBOOK ENTRIES (the load-bearing gather coverage).
//   `e` is the UE4M3 scale code applied to ALL four sub-blocks.
static void fill_nvfp4_edge(uint8_t *buf, int nb, int xmode, uint8_t e) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * NVFP4_STRIDE;
    for (int s = 0; s < N_SUB; s++)
      b[s] = e;
    for (int j = 0; j < QK / 2; j++) {
      uint8_t v;
      if (xmode == 0) v = 0x00;
      else if (xmode == 1) v = 0xFF;
      else { uint8_t nib = (uint8_t)(j & 0xF); v = (uint8_t)(nib | (nib << 4)); }
      b[4 + j] = v;
    }
  }
}

static void fill_q8_0_edge(uint8_t *buf, int q8nb, int mode) {
  for (int i = 0; i < q8nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8_STRIDE;
    put_fp16(b + 0);
    for (int j = 0; j < 32; j++)
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
  int q8nb = n / 32; // = 2*nb  (the buffer-sizing fix)
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
  fill_nvfp4(vx, nb);
  fill_q8_0(vy, q8nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int xmode, uint8_t e, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK;
  int q8nb = n / 32;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
  fill_nvfp4_edge(vx, nb, xmode, e);
  fill_q8_0_edge(vy, q8nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {64, 128, 192, 256, 320, 512, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-40 nvfp4 byte-exact (oracle = _generic = the real board kernel, "
         "no arch/riscv nvfp4 exists) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }

  // EDGE: the marching-nibble pattern (all 16 FP4 codebook entries) + the all-0 /
  // all-15 nibbles, crossed with the FULL 256-code UE4M3 scale range (esp. the two
  // specials 0x00/0x7F -> 0, the exp==0 denormals 0x01-0x07, the large normals
  // 0x78-0x7E) + q8 saturation. This is the strongest combined probe of the new
  // UE4M3 decode + the codebook gather.
  for (int e = 0; e < 256; ++e) {
    total += 3;
    char tag[64];
    snprintf(tag, sizeof tag, "all16 e=0x%02X q8=+/-", e);
    fails += check_edge(256, 2, (uint8_t)e, 1, tag, &generic_delta);
    snprintf(tag, sizeof tag, "all16 e=0x%02X q8=+127", e);
    fails += check_edge(256, 2, (uint8_t)e, 0, tag, &generic_delta);
    snprintf(tag, sizeof tag, "nib15 e=0x%02X q8=+/-", e);
    fails += check_edge(192, 1, (uint8_t)e, 1, tag, &generic_delta);
  }
  printf("checked %d cases, %d failures (vs ggml's RVV transcription)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING FP4 codebook proof): same input bytes, but
  // the reference uses the WRONG (linear nibble-8) codebook. Our kernel embeds the
  // real kvalues_mxfp4 table -> it MUST diverge.
  {
    int n = 2048, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    // Random nibbles + q8 all +127 (NON-ZERO integer sum). Pin a tame UE4M3 scale
    // (0x38 ~ 1.0) so the products are fp32-representable.
    fill_nvfp4(vx, nb);
    for (int i = 0; i < nb; i++)
      for (int s = 0; s < N_SUB; s++) vx[(size_t)i * NVFP4_STRIDE + s] = 0x38;
    fill_q8_0_edge(vy, q8nb, 0);
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_tab(n, &s_real, vx, vy, kvalues_mxfp4, ggml_ue4m3_to_fp32);
    ggml_generic_tab(n, &s_wrong, vx, vy, kvalues_linear, ggml_ue4m3_to_fp32);
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

  // Negative control 2 (the LOAD-BEARING UE4M3 HALF-scale proof): same input bytes,
  // but the reference drops the *0.5f (the FULL UE4M3 decode). Our kernel embeds the
  // half form -> it MUST diverge from the full-scale reference (exactly 2x).
  {
    int n = 2048, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    fill_nvfp4(vx, nb);
    for (int i = 0; i < nb; i++)
      for (int s = 0; s < N_SUB; s++) vx[(size_t)i * NVFP4_STRIDE + s] = 0x38;
    fill_q8_0_edge(vy, q8nb, 0);
    float so = 0, s_half = 0, s_full = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_tab(n, &s_half, vx, vy, kvalues_mxfp4, ggml_ue4m3_to_fp32);
    ggml_generic_tab(n, &s_full, vx, vy, kvalues_mxfp4, ggml_ue4m3_to_fp32_full);
    if (std::memcmp(&so, &s_half, 4) != 0) {
      printf("NEG-CTRL-2 setup FAILED: ours != half-scale ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_full, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: WRONG (full, no *0.5f) UE4M3 scale still "
             "matched (vacuous -- the half scale is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: WRONG (full, no *0.5f) UE4M3 scale diverges "
             "(ours=%.9g half=%.9g full=%.9g) -> the UE4M3 HALF scale is load-bearing\n",
             so, s_half, s_full);
    }
    free(vx);
    free(vy);
  }

  // Negative control 3: flip ONE nibble of vx; our kernel must DIFFER from a fresh
  // ggml run on the UNperturbed data (proves the check is non-vacuous: the FP4
  // nibbles are actually consumed).
  {
    int n = 256, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    fill_nvfp4(vx, nb);
    for (int i = 0; i < nb; i++)
      for (int s = 0; s < N_SUB; s++) vx[(size_t)i * NVFP4_STRIDE + s] = 0x38;
    fill_q8_0_edge(vy, q8nb, 0);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[4] ^= 0x07; // flip part of the low nibble of sub-block 0's first byte (@+4)
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
    printf("ALL nvfp4 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("nvfp4 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
