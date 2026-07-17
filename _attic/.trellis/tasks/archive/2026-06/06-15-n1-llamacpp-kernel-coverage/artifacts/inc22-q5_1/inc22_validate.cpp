// INC-22 q5_1 x q8_1 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q5_1_q8_1_block_dot op
// (the COMPLETE ggml ggml_vec_dot_q5_1_q8_1 block kernel: AoS QK=32 block loop +
// per-block dual fp16 scale + MIN/SUM correction + 5-bit UNSIGNED weight
// reconstruction + fp32 accumulation + *s store) computes the SAME fp32 result
// `*s` as ggml's OWN q5_1 _generic kernel, BIT-FOR-BIT (memcmp of the float bits),
// over random block_q5_1 x block_q8_1 arrays at n in {32,...,8192} + named edge
// cases (q5 all 0/31, the qh bit pattern, q8 +/-127, the MIN term). TWO negative
// controls (a perturbed qh bit AND a perturbed m_x min scale) must FAIL.
//
// q5_1 is Family B (scale+MIN, asymmetric) with a 5-BIT UNSIGNED weight -- it
// COMBINES q5_0's 5-bit reconstruction (a 4-bit nibble PLUS a per-element 5th high
// bit from a separate 32-bit qh field) and q4_1's scale+min fold. The weight is an
// UNSIGNED q5 in [0,31] (NO -16 offset; the bias lives in the per-block MIN scale).
// The fold is q4_1's: sumf += (d_x*d_y)*sumi + m_x*s_y, where sumi = sum(q5*q8) and
// s_y is the precomputed scaled activation sum (d_y * sum(q8)).
//
// REFERENCE: a faithful copy of ggml's q5_1 _generic (quants.c:357-398): the
// UNSIGNED 5-bit reconstruction `(qs&0xF)|xh_0`, `(qs>>4)|xh_1`, with
// xh_0 = ((qh>>j)<<4)&0x10 and xh_1 = ((qh>>(j+12)))&0x10, and the EXACT fold
// `sumf += (d_x*d_y)*sumi + m_x*s_y`. The s_y is computed exactly as ggml's
// quantize_row_q8_1 does (s = d * sum(q8)). FP16->FP32 mirrors the board's scalar
// _Float16 widening (lossless). Whole TU built with the same -ffp-contract setting.
//
// The kernel under test is the UNMODIFIED, compiler-emitted q5_1 kernel
// (q5_1_<profile>.cpp) -- every line tagged source_op=tcrv_rvv.q5_1_q8_1_block_dot,
// shape stamped by --tcrv-rvv-materialize-q5-1-schedule (or default mf4 robust).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_q5_1 = { fp16 d; fp16 m; uint8_t qh[4]; uint8_t qs[16]; } sizeof 24
//                d@0, m@2, qh@4, quants@8
//   block_q8_1 = { fp16 d; fp16 s; int8_t qs[32]; }                 sizeof 36
//                d@0, s@2, quants@4
static const int QK = 32;
static const int Q5_STRIDE = 24;
static const int Q8_STRIDE = 36;

// ---- board fp16 <-> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) ------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}
static inline void put_fp16_value(uint8_t *p, float v) {
  _Float16 d = (_Float16)v;
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
// 4-arg ABI as exported (n, s, vx, vy).
extern "C" void
tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's REAL q5_1 RVV kernel (arch/riscv/quants.c:382-433, intrinsic-for-
// intrinsic). The PRIMARY external reference -- the actual kernel the board's
// llama.cpp runs. Note the 5th bit is OR'd in (vor_vx_i8m2_mu, 0x10) with NO -16
// bias (unsigned q5), and the fold is the q4_1 scale+min statement.
static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  const int qk = QK;
  const int nb = n / qk;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  int ib = 0;
  float sumf = 0;
  size_t vl;
  size_t vlenb = __riscv_vlenb();
  for (; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * Q5_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 8;          // q5 nibbles
    const uint8_t *xqh = xb + 4;          // qh field
    const int8_t *yqs = (const int8_t *)(yb + 4);
    uint16_t xdh, xmh, ydh, ysh;
    std::memcpy(&xdh, xb + 0, 2);
    std::memcpy(&xmh, xb + 2, 2);
    std::memcpy(&ydh, yb + 0, 2);
    std::memcpy(&ysh, yb + 2, 2);

    vl = qk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
    vint8m2_t v0c;
    if (vlenb == 16) {
      v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
    } else {
      v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
      v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
    }

    vl = qk;
    vbool4_t qh = __riscv_vlm_v_b4(xqh, vl);
    vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi +
            fp16_to_fp32(xmh) * fp16_to_fp32(ysh);
  }
  *s = sumf;
}

// ---- ggml's q5_1 _generic (quants.c:357-398, the byte-exact same math) -------
static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  const int qk = QK;
  const int nb = n / qk;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  int ib = 0;
  float sumf = 0;
  for (; ib < nb; ++ib) {
    const uint8_t *xb = x + ib * Q5_STRIDE;
    const uint8_t *yb = y + ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 8;          // q5 nibbles
    const int8_t *yqs = (const int8_t *)(yb + 4);
    uint16_t xdh, xmh, ydh, ysh;
    std::memcpy(&xdh, xb + 0, 2);
    std::memcpy(&xmh, xb + 2, 2);
    std::memcpy(&ydh, yb + 0, 2);
    std::memcpy(&ysh, yb + 2, 2);
    uint32_t qh;
    std::memcpy(&qh, xb + 4, sizeof(qh));

    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; j++) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
      const int32_t x0 = (xqs[j] & 0x0F) | xh_0;  // unsigned q5 [0,31]
      const int32_t x1 = (xqs[j] >> 4) | xh_1;
      sumi0 += x0 * yqs[j];
      sumi1 += x1 * yqs[j + qk / 2];
    }
    int sumi = sumi0 + sumi1;
    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi +
            fp16_to_fp32(xmh) * fp16_to_fp32(ysh);
  }
  *s = sumf;
}

// ---- random block_q5_1 / block_q8_1 array generators -------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  put_fp16_value(p, ((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
}

static void fill_q5_1(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q5_STRIDE;
    put_fp16(b + 0);                       // d
    put_fp16(b + 2);                       // m (the MIN scale)
    for (int k = 0; k < 4; k++)
      b[4 + k] = (uint8_t)(xrand() % 256); // qh (5th-bit field)
    for (int j = 0; j < QK / 2; j++)
      b[8 + j] = (uint8_t)(xrand() % 256); // packed nibbles
  }
}

// q8_1's s field = d * sum(q8); fill the quants then set s exactly as ggml's
// quantize_row_q8_1 does (so the MIN term m_x*s_y is byte-exact).
static void fill_q8_1(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q8_STRIDE;
    put_fp16(b + 0); // d
    int sum = 0;
    for (int j = 0; j < QK; j++) {
      int8_t q = (int8_t)(xrand() % 256);
      b[4 + j] = (uint8_t)q;
      sum += q;
    }
    uint16_t dh;
    std::memcpy(&dh, b + 0, 2);
    put_fp16_value(b + 2, fp16_to_fp32(dh) * (float)sum); // s = d * sum(q8)
  }
}

// Edge-case fillers: pin the weight to a known 5-bit extreme + a qh bit pattern.
//   mode 0: all nibbles 0x00, all qh bits 0 -> every q5 = 0.
//   mode 1: all nibbles 0xFF, all qh bits 1 -> every q5 = 31.
//   mode 2: alternating qh bits, full nibble range (a strong qh-alignment probe).
static void fill_q5_1_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q5_STRIDE;
    put_fp16(b + 0);
    put_fp16(b + 2);
    uint32_t qh;
    if (mode == 0) qh = 0x00000000u;
    else if (mode == 1) qh = 0xFFFFFFFFu;
    else qh = 0xAAAAAAAAu;
    std::memcpy(b + 4, &qh, 4);
    for (int j = 0; j < QK / 2; j++) {
      if (mode == 0) b[8 + j] = 0x00;
      else if (mode == 1) b[8 + j] = 0xFF;
      else b[8 + j] = (uint8_t)((j * 17) & 0xFF);
    }
  }
}

// q8 +/-127 saturation edge; recompute s = d * sum(q8) for the MIN term.
static void fill_q8_1_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q8_STRIDE;
    put_fp16(b + 0);
    int sum = 0;
    for (int j = 0; j < QK; j++) {
      int8_t q = (int8_t)((mode == 0) ? 127 : ((j & 1) ? -128 : 127));
      b[4 + j] = (uint8_t)q;
      sum += q;
    }
    uint16_t dh;
    std::memcpy(&dh, b + 0, 2);
    put_fp16_value(b + 2, fp16_to_fp32(dh) * (float)sum);
  }
}

#ifndef CONTRACT_REFS
#define CONTRACT_REFS 1
#endif

static int check_buffers(int n, uint8_t *vx, uint8_t *vy, const char *tag,
                         int *generic_delta) {
  float so = 0, sg = 0, sr = 0;
  our_kernel(n, &so, vx, vy);
  ggml_rvv(n, &sg, vx, vy);       // PRIMARY: the REAL ggml RVV q5_1 kernel
  ggml_generic(n, &sr, vx, vy);   // the byte-exact integer-then-fold definition
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
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_q5_1(vx, nb);
  fill_q8_1(vy, nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int xmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_q5_1_edge(vx, nb, xmode);
  fill_q8_1_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-22 q5_1 byte-exact (PRIMARY target: ggml's REAL RVV kernel) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }
  // Edge cases: q5 all 0 (mode0), all 31 (mode1), qh bit pattern (mode2);
  // crossed with q8 +127 (ymode0) and q8 +/-128/+127 (ymode1).
  for (int n : n_set) {
    total += 6;
    fails += check_edge(n, 0, 0, "q5=0 x q8=+127", &generic_delta);
    fails += check_edge(n, 1, 0, "q5=31 x q8=+127", &generic_delta);
    fails += check_edge(n, 1, 1, "q5=31 x q8=+/-", &generic_delta);
    fails += check_edge(n, 0, 1, "q5=0 x q8=+/-", &generic_delta);
    fails += check_edge(n, 2, 1, "qh-pattern x q8+/-", &generic_delta);
    fails += check_edge(n, 2, 0, "qh-pattern x q8+127", &generic_delta);
  }
  printf("checked %d cases, %d failures (vs ggml's REAL RVV kernel)\n", total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1: flip ONE qh bit of vx; our kernel must DIFFER from a fresh
  // ggml-rvv run on the UNperturbed data (proves the 5th-bit field is consumed).
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_q5_1(vx, nb);
    fill_q8_1(vy, nb);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[4] ^= 0x01; // flip qh bit 0 of block 0 (element 0's 5th bit)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL (qh) FAILED: perturbed qh still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control OK (qh): flipped qh 5th-bit diverges (qh consumed)\n");
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the q5_1 MIN-term control): perturb the per-block m_x min
  // scale; our kernel must DIFFER from a fresh _generic run on the UNperturbed
  // data (proves the m_x*s_y MIN term is actually folded -- a Family-B-specific
  // discrimination q5_0 cannot have). s_y is left nonzero by construction.
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_q5_1(vx, nb);
    fill_q8_1(vy, nb);
    // Force block 0's m_x and s_y nonzero so the MIN term is materially present.
    put_fp16_value(vx + 2, 0.5f);    // m_x = 0.5
    // ensure s_y != 0: recompute block-0 s from its q8 sum (already done in fill,
    // but pin a nonzero d so s != 0 even if the sum happened to be 0).
    {
      int sum = 0;
      for (int j = 0; j < QK; j++) sum += (int8_t)vy[4 + j];
      if (sum == 0) { vy[4] = (uint8_t)(int8_t)5; sum = 5; }
      put_fp16_value(vy + 0, 0.25f);                   // d_y = 0.25
      put_fp16_value(vy + 2, 0.25f * (float)sum);      // s_y = d_y * sum
    }
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    put_fp16_value(vx + 2, 1.5f);    // perturb m_x: 0.5 -> 1.5
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL (min) FAILED: perturbed m_x still matched (vacuous "
             "-- the m_x*s_y MIN term is NOT folded)\n");
      fails += 1;
    } else {
      printf("negative control OK (min): perturbed m_x diverges (the m_x*s_y MIN "
             "term is folded; the Family-B fold is non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL q5_1 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("q5_1 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
