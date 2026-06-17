// INC-32 iq4_xs x q8_K byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq4_xs_q8_k_block_dot op
// (the COMPLETE ggml ggml_vec_dot_iq4_xs_q8_K super-block kernel: AoS QK_K=256
// super-block loop + per-super-block d4d8 scale + per-sub-block SIGNED 6-bit scale +
// NON-LINEAR codebook decode + FLOAT-domain fold + *s store) computes the SAME fp32
// result `*s` as ggml's OWN _generic iq4_xs reference, BIT-FOR-BIT (memcmp of the
// float bits), over random block_iq4_xs x block_q8_K arrays at n multiples of 256 +
// named edge cases (all 16 codebook entries exercised, the scale extremes ls=-32 /
// ls=+31, q8 +/-127). TWO negative controls (a WRONG codebook + a WRONG scale bias)
// must FAIL -> proves the codebook AND the scales_l/scales_h extraction are the live,
// load-bearing mechanisms.
//
// iq4_xs is the SUPER-BLOCK variant of iq4_nl: it REUSES the SAME 16-entry NON-LINEAR
// int8 codebook (kvalues_iq4nl[16]) inside the q4_K-style super-block structure (the
// q8_K activation + a per-sub-block SIGNED 6-bit scale from scales_l[4]+scales_h,
// biased -32). There is NO min term (symmetric, like q6_K). The fold is in the FLOAT
// domain (d1 = d4d8*(ls-32) then sumf += d1*sumi) -- the byte-exactness pivot at
// -ffp-contract=off (TWO separate float roundings, NOT an integer aux32 deferral).
//
// REFERENCE: ggml's VERBATIM _generic (quants.c:1232-1276): the ib+=2 pair loop with
// the progressive h>>=4 scale extraction, named d1/d2, qs+=16, q8+=32 -- NOT a flat
// paraphrase, so a match independently re-proves the closed-form scale equivalence on
// the actual test bytes. FP16->FP32 mirrors the board's scalar _Float16 widening
// (lossless). Whole TU built with the same -ffp-contract setting.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq4_xs kernel
// (iq4_xs_emitted.cpp) -- every line tagged source_op=tcrv_rvv.iq4_xs_q8_k_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#define QK_K 256
static const int IQ4XS_STRIDE = 136; // sizeof block_iq4_xs
static const int Q8K_STRIDE = 292;   // sizeof block_q8_K

// ---- block formats (ggml-common.h, QK_K = 256) -------------------------------
//   block_iq4_xs (136 bytes): d(fp16) @0 | scales_h(u16) @2 | scales_l[4] @4 |
//                             qs[128] @8
//   block_q8_K   (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260 (unused)
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } block_iq4_xs;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;

// ---- ggml's 16-entry NON-LINEAR codebook (ggml-common.h:1110-1112) -----------
// The SAME table iq4_nl uses (iq4_xs REUSES it).
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                         -10,  1,    13,  25,  38,  53,  69,
                                         89,   113};
// WRONG codebook for negative control 1: the LINEAR nibble-8 decode (q4_0's). If
// the codebook were NOT load-bearing, swapping it would not change the result.
static const int8_t kvalues_linear[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7};

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
extern "C" void
tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_ggml_vec_dot_iq4_xs_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_ggml_vec_dot_iq4_xs_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's VERBATIM _generic (quants.c:1232-1276) ---------------------------
// Parameterised by the codebook table AND the scale bias so the negative controls
// can swap them. The structure is ggml's EXACT ib+=2 pair loop with progressive
// h>>=4 (NOT a flat paraphrase).
static void ggml_generic_param(int n, float *s, const void *vx, const void *vy,
                               const int8_t *table, int scale_bias) {
  const block_iq4_xs *x = (const block_iq4_xs *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;
  const int nb = n / QK_K;
  float sumf = 0;
  for (int ibl = 0; ibl < nb; ++ibl) {
    const float d4d8 = fp16_to_fp32(x[ibl].d) * y[ibl].d;
    uint16_t h = x[ibl].scales_h;
    const uint8_t *qs = x[ibl].qs;
    const int8_t *q8 = y[ibl].qs;
    for (int ib = 0; ib < QK_K / 32; ib += 2) {
      const uint8_t ls1 = (x[ibl].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30);
      const uint8_t ls2 = (x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30);
      h >>= 4;
      const float d1 = d4d8 * (ls1 - scale_bias);
      const float d2 = d4d8 * (ls2 - scale_bias);
      int sumi1 = 0, sumi2 = 0;
      for (int j = 0; j < 16; ++j) {
        sumi1 += q8[j + 0] * table[qs[j] & 0xf];
        sumi2 += q8[j + 16] * table[qs[j] >> 4];
      }
      sumf += d1 * (sumi1 + sumi2);
      qs += 16;
      q8 += 32;
      sumi1 = sumi2 = 0;
      for (int j = 0; j < 16; ++j) {
        sumi1 += q8[j + 0] * table[qs[j] & 0xf];
        sumi2 += q8[j + 16] * table[qs[j] >> 4];
      }
      sumf += d2 * (sumi1 + sumi2);
      qs += 16;
      q8 += 32;
    }
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_param(n, s, vx, vy, kvalues_iq4nl, 32);
}

// ---- random block_iq4_xs / block_q8_K array generators -----------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_iq4_xs(uint8_t *buf, int nbl) {
  for (int i = 0; i < nbl; i++) {
    uint8_t *b = buf + (size_t)i * IQ4XS_STRIDE;
    put_fp16(b + 0);                          // d
    b[2] = (uint8_t)(xrand() % 256);          // scales_h lo
    b[3] = (uint8_t)(xrand() % 256);          // scales_h hi
    for (int j = 0; j < 4; j++)
      b[4 + j] = (uint8_t)(xrand() % 256);    // scales_l[4]
    for (int j = 0; j < 128; j++)
      b[8 + j] = (uint8_t)(xrand() % 256);    // qs (every nibble value)
  }
}

static void fill_q8_K(uint8_t *buf, int nbl) {
  for (int i = 0; i < nbl; i++) {
    uint8_t *b = buf + (size_t)i * Q8K_STRIDE;
    float dy = (float)((int)(xrand() % 2000) - 1000) / 4096.0f; // fp32 activation d
    std::memcpy(b + 0, &dy, 4);
    for (int j = 0; j < 256; j++)
      b[4 + j] = (uint8_t)(int8_t)(xrand() % 256);
    for (int j = 0; j < 32; j++)
      b[260 + j] = 0; // bsums (unused -- no min term)
  }
}

// Edge-case fillers.
//   xmode 0: nibbles all 0x00 -> entry 0 (-127); scales_l/scales_h all 0x00 -> ls=-32.
//   xmode 1: nibbles all 0xFF -> entry 15 (+113); scales_l/scales_h all 0xFF -> ls=+31.
//   xmode 2: marching nibble j&0xF (ALL 16 codebook entries); scales random.
//   xmode 3: marching nibble (all 16 entries) + scale extremes per sub-block
//            (alternating ls=-32 / ls=+31 via scales_l/scales_h all 0x00 / 0xFF).
static void fill_iq4_xs_edge(uint8_t *buf, int nbl, int mode) {
  for (int i = 0; i < nbl; i++) {
    uint8_t *b = buf + (size_t)i * IQ4XS_STRIDE;
    put_fp16(b + 0);
    if (mode == 0) {
      b[2] = b[3] = 0x00;                   // scales_h = 0 -> high bits 0
      for (int j = 0; j < 4; j++) b[4 + j] = 0x00; // scales_l = 0 -> low nibble 0; ls=0 -> -32
      for (int j = 0; j < 128; j++) b[8 + j] = 0x00;
    } else if (mode == 1) {
      b[2] = b[3] = 0xFF;                   // scales_h = 0xFFFF -> high 2 bits set
      for (int j = 0; j < 4; j++) b[4 + j] = 0xFF; // scales_l = 0xFF -> low nibble 0xF; ls=63 -> +31
      for (int j = 0; j < 128; j++) b[8 + j] = 0xFF;
    } else if (mode == 2) {
      b[2] = (uint8_t)(xrand() % 256);
      b[3] = (uint8_t)(xrand() % 256);
      for (int j = 0; j < 4; j++) b[4 + j] = (uint8_t)(xrand() % 256);
      for (int j = 0; j < 128; j++) {
        uint8_t nib = (uint8_t)(j & 0xF);
        b[8 + j] = (uint8_t)(nib | (nib << 4)); // marches all 16 entries
      }
    } else { // mode 3: marching nibble + scale extremes (max negative + max positive)
      b[2] = b[3] = 0xFF;                   // all high scale bits set
      for (int j = 0; j < 4; j++) b[4 + j] = (j & 1) ? 0x00 : 0xFF; // alternate extremes
      for (int j = 0; j < 128; j++) {
        uint8_t nib = (uint8_t)(j & 0xF);
        b[8 + j] = (uint8_t)(nib | (nib << 4));
      }
    }
  }
}

// q8 +/-127 saturation edge.
static void fill_q8_K_edge(uint8_t *buf, int nbl, int mode) {
  for (int i = 0; i < nbl; i++) {
    uint8_t *b = buf + (size_t)i * Q8K_STRIDE;
    float dy = (float)((int)(xrand() % 2000) - 1000) / 4096.0f;
    std::memcpy(b + 0, &dy, 4);
    for (int j = 0; j < 256; j++)
      b[4 + j] = (uint8_t)(int8_t)((mode == 0) ? 127 : ((j & 1) ? -127 : 127));
    for (int j = 0; j < 32; j++)
      b[260 + j] = 0;
  }
}

#ifndef CONTRACT_REFS
#define CONTRACT_REFS 1
#endif

static int check_buffers(int n, uint8_t *vx, uint8_t *vy, const char *tag,
                         int *generic_delta) {
  float so = 0, sr = 0;
  our_kernel(n, &so, vx, vy);
  ggml_generic(n, &sr, vx, vy);
  int fail = 0;
  if (std::memcmp(&so, &sr, 4) != 0) {
    *generic_delta += 1;
    if (CONTRACT_REFS) {
      printf("  %-26s n=%-7d FAIL vs _generic : ours=%.9g gen=%.9g\n", tag, n, so, sr);
      fail = 1;
    }
  }
  return fail;
}

static int check_random(int n, int *generic_delta) {
  int nbl = n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nbl * IQ4XS_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nbl * Q8K_STRIDE);
  fill_iq4_xs(vx, nbl);
  fill_q8_K(vy, nbl);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int xmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nbl = n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nbl * IQ4XS_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nbl * Q8K_STRIDE);
  fill_iq4_xs_edge(vx, nbl, xmode);
  fill_q8_K_edge(vy, nbl, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {256, 512, 768, 1024, 1280, 1536, 2048, 4096, 8192, 16384};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-32 iq4_xs byte-exact (target: ggml's VERBATIM _generic iq4_xs) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }
  // Edge cases: scale extremes + all-16-codebook-entries + q8 +/-127.
  for (int n : n_set) {
    total += 7;
    fails += check_edge(n, 0, 0, "ls=-32,nib0 x q8+127",   &generic_delta);
    fails += check_edge(n, 1, 0, "ls=+31,nib15 x q8+127",  &generic_delta);
    fails += check_edge(n, 1, 1, "ls=+31,nib15 x q8+/-",   &generic_delta);
    fails += check_edge(n, 0, 1, "ls=-32,nib0 x q8+/-",    &generic_delta);
    fails += check_edge(n, 2, 1, "all-16-entries x q8+/-", &generic_delta);
    fails += check_edge(n, 2, 0, "all-16-entries x q8+127", &generic_delta);
    fails += check_edge(n, 3, 1, "scale-extremes+all-16 x q8+/-", &generic_delta);
  }
  printf("checked %d cases, %d failures (vs ggml's VERBATIM _generic iq4_xs)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "reference's OWN FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING codebook proof): keep the EXACT SAME input
  // bytes, but compute the reference with the WRONG (linear nibble-8) codebook. Our
  // kernel embeds the real kvalues_iq4nl table, so it MUST diverge.
  {
    int n = 4096, nbl = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nbl * IQ4XS_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nbl * Q8K_STRIDE);
    fill_iq4_xs_edge(vx, nbl, 2); // marching nibble: all 16 entries differ linear vs nl
    fill_q8_K_edge(vy, nbl, 1);   // q8 +/-127 so the products are non-trivial
    float so = 0, s_real = 0, s_linear = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_param(n, &s_real, vx, vy, kvalues_iq4nl, 32);    // real table
    ggml_generic_param(n, &s_linear, vx, vy, kvalues_linear, 32); // WRONG (linear)
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-1 setup FAILED: ours != real-codebook ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_linear, 4) == 0) {
      printf("NEGATIVE CONTROL 1 FAILED: WRONG (linear) codebook still matched "
             "(vacuous -- the codebook is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 1 OK: WRONG (linear nibble-8) codebook diverges "
             "(ours=%.9g real=%.9g linear=%.9g) -> the codebook is load-bearing\n",
             so, s_real, s_linear);
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the LOAD-BEARING scale proof): keep the EXACT SAME input
  // bytes, but compute the reference with the WRONG scale bias (-31 instead of -32).
  // Our kernel embeds the real -32 bias, so it MUST diverge -> the scales_l/scales_h
  // extraction is the live, load-bearing mechanism (not dead).
  {
    int n = 4096, nbl = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nbl * IQ4XS_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nbl * Q8K_STRIDE);
    fill_iq4_xs(vx, nbl);
    fill_q8_K_edge(vy, nbl, 1); // q8 +/-127 so the scale change moves the result
    float so = 0, s_real = 0, s_wrongscale = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_param(n, &s_real, vx, vy, kvalues_iq4nl, 32);        // real bias -32
    ggml_generic_param(n, &s_wrongscale, vx, vy, kvalues_iq4nl, 31);  // WRONG bias -31
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-2 setup FAILED: ours != real-scale ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrongscale, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: WRONG scale bias still matched "
             "(vacuous -- the scale extraction is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: WRONG scale bias (-31) diverges "
             "(ours=%.9g real=%.9g wrong=%.9g) -> the signed scale is load-bearing\n",
             so, s_real, s_wrongscale);
    }
    free(vx);
    free(vy);
  }

  // Negative control 3: flip ONE nibble of vx; our kernel must DIFFER from a fresh
  // run on the UNperturbed data (proves the check is non-vacuous: nibbles consumed).
  {
    int n = 512, nbl = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nbl * IQ4XS_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nbl * Q8K_STRIDE);
    fill_iq4_xs(vx, nbl);
    fill_q8_K_edge(vy, nbl, 0); // q8 = +127 so the perturbed product is non-zero
    float sr = 0;
    ggml_generic(n, &sr, vx, vy);
    vx[8] ^= 0x0F; // flip the low nibble of element 0 in super-block 0 (qs @8)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sr, 4) == 0) {
      printf("NEGATIVE CONTROL 3 FAILED: perturbed nibble still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 3 OK: flipped nibble diverges (the weight nibbles "
             "are consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL iq4_xs BYTE-EXACT CHECKS PASSED\n");
  else
    printf("iq4_xs BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
