// INC-30 iq4_nl x q8_0 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq4_nl_q8_0_block_dot op
// (the COMPLETE ggml ggml_vec_dot_iq4_nl_q8_0 block kernel: AoS QK=32 block loop +
// per-block dual fp16 scale + NON-LINEAR codebook decode + fp32 accumulation + *s
// store) computes the SAME fp32 result `*s` as ggml's OWN iq4_nl RVV kernel,
// BIT-FOR-BIT (memcmp of the float bits), over random block_iq4_nl x block_q8_0
// arrays at n in {32,64,...,8192} + named edge cases (all 16 codebook entries
// exercised, q8 +/-127). TWO negative controls (a WRONG codebook + a perturbed
// nibble) must FAIL -> proves the codebook is the live, load-bearing mechanism.
//
// iq4_nl is Family A (single-scale) and SHARES block_q4_0's byte shape, but the
// 4-bit nibble does NOT decode linearly (`nibble - 8`); it INDEXES a 16-entry
// NON-LINEAR int8 CODEBOOK: value q = kvalues_iq4nl[nibble] (a LOOKUP). The fold is
// iq4_nl's scales-first order: sumf += (float)sumi * (d_x*d_y), where
// d = d_y * d_x. (ggml _generic quants.c:1203-1230; ggml RVV vl256 path
// arch/riscv/quants.c:5534-5589, byte-exact same math as vl128.)
//
// REFERENCE: ggml's _generic AND a faithful copy of ggml's RVV iq4_nl method
// (vand/vsrl nibble split -> vrgather through the kvalues table -> vwmul/vwmacc ->
// vwredsum), both with an order-independent integer sumi and the IDENTICAL fp32
// statement, so all three are the same fp32 result bit-for-bit. FP16->FP32 mirrors
// the board's scalar _Float16 widening (lossless). Whole TU built with the same
// -ffp-contract setting.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq4_nl kernel
// (iq4_nl_m1_elided.cpp) -- every line tagged source_op=tcrv_rvv.iq4_nl_q8_0_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_iq4_nl = { fp16 d; uint8_t qs[16]; }  sizeof 18, nibbles @2
//   block_q8_0   = { fp16 d; int8_t  qs[32]; }  sizeof 34, quants @2
static const int QK = 32;
static const int IQ4_STRIDE = 18;
static const int Q8_STRIDE = 34;

// ---- ggml's 16-entry NON-LINEAR codebook (ggml-common.h:1110-1112) -----------
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                         -10,  1,    13,  25,  38,  53,  69,
                                         89,   113};
// The WRONG codebook for negative control 1: the LINEAR nibble-8 decode (what
// q4_0 uses). If the codebook were NOT load-bearing, swapping it would not change
// the result -- it MUST.
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
tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's REAL iq4_nl RVV method (arch/riscv/quants.c, vl256 split-half) ----
// Parameterised by the codebook table so the negative control can swap it.
static void ggml_rvv_tab(int n, float *s, const void *vx, const void *vy,
                         const int8_t *table) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  vint8m1_t values = __riscv_vle8_v_i8m1(table, 16);
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * IQ4_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 2;                 // packed nibbles
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t xdh, ydh;
    std::memcpy(&xdh, xb + 0, 2);
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

    sumf += (float)sumi * (fp16_to_fp32(xdh) * fp16_to_fp32(ydh));
  }
  *s = sumf;
}

static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  ggml_rvv_tab(n, s, vx, vy, kvalues_iq4nl);
}

// ---- ggml's _generic (quants.c:1203-1230, the byte-exact same math) ----------
// Parameterised by the codebook table (the negative control swaps it).
static void ggml_generic_tab(int n, float *s, const void *vx, const void *vy,
                             const int8_t *table) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * IQ4_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 2;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t xdh, ydh;
    std::memcpy(&xdh, xb + 0, 2);
    std::memcpy(&ydh, yb + 0, 2);
    const float d = fp16_to_fp32(ydh) * fp16_to_fp32(xdh);
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
  ggml_generic_tab(n, s, vx, vy, kvalues_iq4nl);
}

// ---- random block_iq4_nl / block_q8_0 array generators -----------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_iq4_nl(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * IQ4_STRIDE;
    put_fp16(b + 0);                       // d
    for (int j = 0; j < QK / 2; j++)
      b[2 + j] = (uint8_t)(xrand() % 256); // packed nibbles (every nibble value)
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

// Edge-case fillers: pin the weight nibbles to cover EVERY codebook entry.
//   mode 0: all nibbles 0x00       -> both halves index entry 0 (kvalues[0]=-127).
//   mode 1: all nibbles 0xFF       -> both halves index entry 15 (kvalues[15]=+113).
//   mode 2: byte j = (j&0xF)|((j&0xF)<<4) marching the nibble through 0..15 ->
//           EXERCISES ALL 16 CODEBOOK ENTRIES (the load-bearing gather coverage).
static void fill_iq4_nl_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * IQ4_STRIDE;
    put_fp16(b + 0);
    for (int j = 0; j < QK / 2; j++) {
      uint8_t v;
      if (mode == 0) v = 0x00;
      else if (mode == 1) v = 0xFF;
      else { uint8_t nib = (uint8_t)(j & 0xF); v = (uint8_t)(nib | (nib << 4)); }
      b[2 + j] = v;
    }
  }
}

// q8 +/-127 saturation edge.
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
    printf("  %-22s n=%-6d FAIL vs ggml-rvv : ours=%.9g ggml=%.9g\n", tag, n, so, sg);
    fail = 1;
  }
  if (std::memcmp(&so, &sr, 4) != 0) {
    *generic_delta += 1;
    if (CONTRACT_REFS) {
      printf("  %-22s n=%-6d FAIL vs _generic : ours=%.9g gen=%.9g\n", tag, n, so, sr);
      fail = 1;
    }
  }
  return fail;
}

static int check_random(int n, int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * IQ4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_iq4_nl(vx, nb);
  fill_q8_0(vy, nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int xmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * IQ4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_iq4_nl_edge(vx, nb, xmode);
  fill_q8_0_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-30 iq4_nl byte-exact (PRIMARY target: ggml's REAL iq4_nl RVV "
         "method) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }
  // Edge cases: nibble all-0 (entry 0), all-15 (entry 15), and the marching-nibble
  // pattern that exercises ALL 16 codebook entries; crossed with q8 +127 and
  // q8 +/-128/+127. The marching x saturation cases are the strongest codebook-
  // coverage + saturation probe; run them at several n incl. small block counts.
  for (int n : n_set) {
    total += 6;
    fails += check_edge(n, 0, 0, "nib=0 x q8=+127",  &generic_delta);
    fails += check_edge(n, 1, 0, "nib=15 x q8=+127", &generic_delta);
    fails += check_edge(n, 1, 1, "nib=15 x q8=+/-",  &generic_delta);
    fails += check_edge(n, 0, 1, "nib=0 x q8=+/-",   &generic_delta);
    fails += check_edge(n, 2, 1, "all-16-entries x q8+/-", &generic_delta);
    fails += check_edge(n, 2, 0, "all-16-entries x q8+127",&generic_delta);
  }
  printf("checked %d cases, %d failures (vs ggml's REAL iq4_nl RVV method)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING codebook proof): keep the EXACT SAME input
  // bytes, but compute the reference with the WRONG (linear nibble-8) codebook. Our
  // kernel embeds the real kvalues_iq4nl table, so it MUST diverge from a
  // linear-codebook reference -> the codebook is the live mechanism, not dead. (And
  // the linear-codebook reference IS the q4_0 result on those bytes.)
  {
    int n = 2048, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * IQ4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_iq4_nl_edge(vx, nb, 2); // marching nibble: all 16 entries differ linear vs nl
    fill_q8_0_edge(vy, nb, 1);   // q8 +/-127 so the products are non-trivial
    float so = 0, s_linear = 0, s_real = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_tab(n, &s_real, vx, vy, kvalues_iq4nl);   // real table
    ggml_generic_tab(n, &s_linear, vx, vy, kvalues_linear); // WRONG (linear) table
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

  // Negative control 2: flip ONE nibble of vx; our kernel must DIFFER from a fresh
  // ggml run on the UNperturbed data (proves the check is non-vacuous: the weight
  // nibbles are actually consumed).
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * IQ4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_iq4_nl(vx, nb);
    fill_q8_0_edge(vy, nb, 0); // q8 = +127 so the perturbed product is non-zero
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[2] ^= 0x0F; // flip the low nibble of element 0 in block 0
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: perturbed nibble still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: flipped nibble diverges (the weight nibbles "
             "are consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL iq4_nl BYTE-EXACT CHECKS PASSED\n");
  else
    printf("iq4_nl BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
