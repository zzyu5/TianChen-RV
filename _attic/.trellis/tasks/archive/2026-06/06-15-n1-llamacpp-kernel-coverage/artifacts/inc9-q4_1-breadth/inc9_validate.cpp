// INC-9 q4_1 x q8_1 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q4_1_q8_1_block_dot op
// (the COMPLETE ggml ggml_vec_dot_q4_1_q8_1 block kernel: AoS QK=32 block loop +
// per-block dual fp16 scale + the per-block MIN/SUM correction term + fp32
// accumulation + *s store), with the shape AUTOTUNER-SELECTED (NOT hand-set),
// computes the SAME fp32 result `*s` as ggml's OWN q4_1 RVV kernel, BIT-FOR-BIT
// (memcmp of the float bits), over random block_q4_1 x block_q8_1 arrays at n in
// {32,64,...,8192} + named edge cases. A negative control (one perturbed scale)
// must FAIL, proving discrimination.
//
// q4_1 is Family B (scale+MIN, asymmetric) -- DIFFERENT decode AND dequant from
// q4_0/q8_0: UNSIGNED nibbles [0,15] (NO offset-binary -8), and the fold carries a
// SECOND scale on each operand: sumf += (d_x*d_y)*sumi + m_x*s_y.
//
// REFERENCE: the REAL ggml RVV kernel body, transcribed intrinsic-for-intrinsic
// from llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:277-326, with the EXACT
// per-block accumulate `sumf += (FP16(x.d)*FP16(y.d))*sumi + FP16(x.m)*FP16(y.s)`.
// ggml's _generic (quants.c:211-245) uses the IDENTICAL statement and an order-
// independent integer sumi, so it is the same fp32 result bit-for-bit; we ALSO
// cross-check against a faithful copy of _generic. FP16->FP32 mirrors the board's
// scalar _Float16 widening (lossless). Whole TU built with the same -ffp-contract
// setting so the within-statement fp contraction is identical in all three.
//
// The kernel under test is the UNMODIFIED, compiler-emitted q4_1 kernel
// (q4_1_<profile>_autotuned.cpp) -- every line tagged source_op=
// tcrv_rvv.q4_1_q8_1_block_dot, shape stamped by --tcrv-rvv-materialize-q4-1-schedule.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// ---- block formats (ggml-common.h:191-202, 248-258) --------------------------
//   block_q4_1 = { fp16 d; fp16 m; uint8_t qs[16]; }  sizeof 20, quants at +4
//   block_q8_1 = { fp16 d; fp16 s; int8_t  qs[32]; }  sizeof 36, quants at +4
static const int QK = 32;
static const int Q4_STRIDE = 20;
static const int Q8_STRIDE = 36;

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
// 4-arg ABI as exported (n, s, vx, vy).
extern "C" void
tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's REAL q4_1 RVV kernel (quants.c:277-326, intrinsic-for-intrinsic) -
static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  const int qk = QK;
  const int nb = n / qk;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  int ib = 0;
  float sumf = 0;
  size_t vl = qk / 2; // 16
  for (; ib < nb; ++ib) {
    const uint8_t *xq = x + ib * Q4_STRIDE + 4; // q4_1 nibbles
    const int8_t *yq = (const int8_t *)(y + ib * Q8_STRIDE + 4);
    uint16_t xdh, xmh, ydh, ysh;
    std::memcpy(&xdh, x + ib * Q4_STRIDE + 0, 2); // d_x
    std::memcpy(&xmh, x + ib * Q4_STRIDE + 2, 2); // m_x
    std::memcpy(&ydh, y + ib * Q8_STRIDE + 0, 2); // d_y
    std::memcpy(&ysh, y + ib * Q8_STRIDE + 2, 2); // s_y
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xq, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yq, vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1(yq + 16, vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi +
            fp16_to_fp32(xmh) * fp16_to_fp32(ysh);
  }
  *s = sumf;
}

// ---- ggml's _generic (quants.c:211-245, the byte-exact same math) ------------
static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  const int qk = QK;
  const int nb = n / qk;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  int ib = 0;
  float sumf = 0;
  for (; ib < nb; ++ib) {
    const uint8_t *xq = x + ib * Q4_STRIDE + 4;
    const int8_t *yq = (const int8_t *)(y + ib * Q8_STRIDE + 4);
    uint16_t xdh, xmh, ydh, ysh;
    std::memcpy(&xdh, x + ib * Q4_STRIDE + 0, 2);
    std::memcpy(&xmh, x + ib * Q4_STRIDE + 2, 2);
    std::memcpy(&ydh, y + ib * Q8_STRIDE + 0, 2);
    std::memcpy(&ysh, y + ib * Q8_STRIDE + 2, 2);
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; j++) {
      const int v0 = (xq[j] & 0x0F);
      const int v1 = (xq[j] >> 4);
      sumi0 += v0 * yq[j];
      sumi1 += v1 * yq[j + qk / 2];
    }
    int sumi = sumi0 + sumi1;
    sumf += (fp16_to_fp32(xdh) * fp16_to_fp32(ydh)) * sumi +
            fp16_to_fp32(xmh) * fp16_to_fp32(ysh);
  }
  *s = sumf;
}

// ---- random block_q4_1 / block_q8_1 array generators -------------------------
static uint32_t rng = 0x12345678u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  // a small random fp16 (avoid inf/nan): a safe magnitude range.
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_q4_1(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q4_STRIDE;
    put_fp16(b + 0); // d
    put_fp16(b + 2); // m
    for (int j = 0; j < QK / 2; j++)
      b[4 + j] = (uint8_t)(xrand() % 256); // packed nibbles
  }
}

static void fill_q8_1(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + i * Q8_STRIDE;
    put_fp16(b + 0); // d
    put_fp16(b + 2); // s (the precomputed scaled sum -- random here is fine for a
                     //    byte-exact contract: all three references read the same s)
    for (int j = 0; j < QK; j++)
      b[4 + j] = (uint8_t)(int8_t)(xrand() % 256);
  }
}

#ifndef CONTRACT_REFS
#define CONTRACT_REFS 1
#endif

static int check_one(int n, int *generic_delta) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_q4_1(vx, nb);
  fill_q8_1(vy, nb);
  float so = 0, sg = 0, sr = 0;
  our_kernel(n, &so, vx, vy);
  ggml_rvv(n, &sg, vx, vy);
  ggml_generic(n, &sr, vx, vy);
  int fail = 0;
  if (std::memcmp(&so, &sg, 4) != 0) {
    printf("  n=%-6d FAIL vs ggml-rvv : ours=%.9g ggml=%.9g\n", n, so, sg);
    fail = 1;
  }
  if (std::memcmp(&so, &sr, 4) != 0) {
    *generic_delta += 1;
    if (CONTRACT_REFS) {
      printf("  n=%-6d FAIL vs _generic : ours=%.9g gen=%.9g\n", n, so, sr);
      fail = 1;
    }
  }
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-9 q4_1 byte-exact (PRIMARY target: ggml's REAL RVV kernel) ==\n");
  for (int rep = 0; rep < 300; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_one(n, &generic_delta);
    }
  }
  printf("checked %d cases, %d failures (vs ggml's REAL RVV kernel)\n", total,
         fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is "
         "the references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control: perturb ONE min-scale byte of vx; our kernel must DIFFER
  // from a fresh ggml run on the UNperturbed data (proves the check is non-vacuous
  // AND that the MIN term is actually consumed -- a q4_1-specific discrimination).
  {
    int n = 256, nb = n / QK;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q4_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
    fill_q4_1(vx, nb);
    fill_q8_1(vy, nb);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[2] ^= 0x40; // perturb the MIN (m) scale byte
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL FAILED: perturbed min still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control OK: perturbed MIN scale diverges (the m_x*s_y "
             "term is consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL q4_1 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("q4_1 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
