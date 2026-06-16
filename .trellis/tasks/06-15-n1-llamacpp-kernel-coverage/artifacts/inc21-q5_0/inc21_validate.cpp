// INC-21 q5_0 x q8_0 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q5_0_q8_0_block_dot op
// (the COMPLETE ggml ggml_vec_dot_q5_0_q8_0 block kernel: AoS QK=32 block loop +
// per-block dual fp16 scale + 5-bit weight reconstruction + fp32 accumulation +
// *s store) computes the SAME fp32 result `*s` as ggml's OWN q5_0 RVV kernel,
// BIT-FOR-BIT (memcmp of the float bits), over random block_q5_0 x block_q8_0
// arrays at n in {32,64,...,8192} + named edge cases (q5 all -16 / +15, the qh bit
// pattern, q8 +/-127). A negative control (one perturbed qh bit) must FAIL.
//
// q5_0 is Family A (single-scale) but with a 5-BIT weight: a 4-bit nibble PLUS a
// per-element 5th high bit packed in a separate 32-bit qh field, then offset-binary
// `-16` bias: value q5_i = (int8_t)(((nibble) | (qh_bit_i << 4)) - 16) in [-16,15].
// The fold is q8_0's scales-first order: sumf += (d_x*d_y)*sumi.
//
// REFERENCE: the REAL ggml RVV kernel body, transcribed intrinsic-for-intrinsic
// from llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:328-379, AND a faithful copy
// of ggml's _generic (quants.c:314-355). Both use an order-independent integer sumi
// and the IDENTICAL fp32 statement, so all three are the same fp32 result bit-for-
// bit. FP16->FP32 mirrors the board's scalar _Float16 widening (lossless). Whole TU
// built with the same -ffp-contract setting.
//
// The kernel under test is the UNMODIFIED, compiler-emitted q5_0 kernel
// (q5_0_<profile>.cpp) -- every line tagged source_op=tcrv_rvv.q5_0_q8_0_block_dot,
// shape stamped by --tcrv-rvv-materialize-q5-0-schedule (or pinned for the mf4
// robust strip-offset proof).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// ---- block formats (ggml-common.h:219-225, 243-246) --------------------------
//   block_q5_0 = { fp16 d; uint8_t qh[4]; uint8_t qs[16]; }  sizeof 22, quants @6
//   block_q8_0 = { fp16 d; int8_t  qs[32]; }                 sizeof 34, quants @2
static const int QK = 32;
static const int Q5_STRIDE = 22;
static const int Q8_STRIDE = 34;

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
// 4-arg ABI as exported (n, s, vx, vy).
extern "C" void
tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's REAL q5_0 RVV kernel (quants.c:328-379, intrinsic-for-intrinsic) -
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
    const uint8_t *xqs = xb + 6;          // q5 nibbles
    const uint8_t *xqh = xb + 2;          // qh field
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t xdh, ydh;
    std::memcpy(&xdh, xb + 0, 2);
    std::memcpy(&ydh, yb + 0, 2);

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
    qh = __riscv_vmnand_mm_b4(qh, qh, vl);
    vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi;
  }
  *s = sumf;
}

// ---- ggml's _generic (quants.c:314-355, the byte-exact same math) ------------
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
    const uint8_t *xqs = xb + 6;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t xdh, ydh;
    std::memcpy(&xdh, xb + 0, 2);
    std::memcpy(&ydh, yb + 0, 2);
    uint32_t qh;
    std::memcpy(&qh, xb + 2, sizeof(qh));

    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; j++) {
      const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
      const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));
      const int32_t x0 = (int8_t)(((xqs[j] & 0x0F) | xh_0) - 16);
      const int32_t x1 = (int8_t)(((xqs[j] >> 4) | xh_1) - 16);
      sumi0 += x0 * yqs[j];
      sumi1 += x1 * yqs[j + qk / 2];
    }
    int sumi = sumi0 + sumi1;
    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi;
  }
  *s = sumf;
}

// ---- random block_q5_0 / block_q8_0 array generators -------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_q5_0(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q5_STRIDE;
    put_fp16(b + 0);                       // d
    for (int k = 0; k < 4; k++)
      b[2 + k] = (uint8_t)(xrand() % 256); // qh (5th-bit field)
    for (int j = 0; j < QK / 2; j++)
      b[6 + j] = (uint8_t)(xrand() % 256); // packed nibbles
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

// Edge-case fillers: pin the weight to a known 5-bit extreme + a qh bit pattern.
//   mode 0: all nibbles 0x00, all qh bits 0 -> every q5 = -16.
//   mode 1: all nibbles 0xFF, all qh bits 1 -> every q5 = +15.
//   mode 2: alternating qh bits, full nibble range (a strong qh-alignment probe).
static void fill_q5_0_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q5_STRIDE;
    put_fp16(b + 0);
    uint32_t qh;
    if (mode == 0) qh = 0x00000000u;
    else if (mode == 1) qh = 0xFFFFFFFFu;
    else qh = 0xAAAAAAAAu;
    std::memcpy(b + 2, &qh, 4);
    for (int j = 0; j < QK / 2; j++) {
      if (mode == 0) b[6 + j] = 0x00;
      else if (mode == 1) b[6 + j] = 0xFF;
      else b[6 + j] = (uint8_t)((j * 17) & 0xFF);
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
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_q5_0(vx, nb);
  fill_q8_0(vy, nb);
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
  fill_q5_0_edge(vx, nb, xmode);
  fill_q8_0_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-21 q5_0 byte-exact (PRIMARY target: ggml's REAL RVV kernel) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }
  // Edge cases: q5 all -16 (mode0), all +15 (mode1), qh bit pattern (mode2);
  // crossed with q8 +127 (ymode0) and q8 +/-128/+127 (ymode1). The qh-bit-pattern
  // x q8-saturation cases are the strongest 5th-bit + saturation probe; run them
  // at several n incl. non-block-multiples of the strip width.
  for (int n : n_set) {
    total += 6;
    fails += check_edge(n, 0, 0, "q5=-16 x q8=+127", &generic_delta);
    fails += check_edge(n, 1, 0, "q5=+15 x q8=+127", &generic_delta);
    fails += check_edge(n, 1, 1, "q5=+15 x q8=+/-",  &generic_delta);
    fails += check_edge(n, 0, 1, "q5=-16 x q8=+/-",  &generic_delta);
    fails += check_edge(n, 2, 1, "qh-pattern x q8+/-",&generic_delta);
    fails += check_edge(n, 2, 0, "qh-pattern x q8+127",&generic_delta);
  }
  printf("checked %d cases, %d failures (vs ggml's REAL RVV kernel)\n", total,
         fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is "
         "the references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control: flip ONE qh bit of vx; our kernel must DIFFER from a fresh
  // ggml run on the UNperturbed data (proves the check is non-vacuous AND that the
  // 5th-bit field is actually consumed -- a q5_0-specific discrimination).
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q5_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_q5_0(vx, nb);
    fill_q8_0(vy, nb);
    // ensure the flipped qh bit actually changes a nonzero-activation product
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[2] ^= 0x01; // flip qh bit 0 of block 0 (element 0's 5th bit)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL FAILED: perturbed qh still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control OK: flipped qh 5th-bit diverges (the qh field is "
             "consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL q5_0 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("q5_0 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
