// INC-5 shape-knob bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our COMPILER emits for each of the three bounded shapes of
// tcrv_rvv.q4_0_q8_0_block_dot --
//   mb1-robust  (multi_block_factor 1, strip_elision robust, integer_core m1)
//   mb2-robust  (multi_block_factor 2, strip_elision robust, integer_core m1)
//   mb4-elided  (multi_block_factor 4, strip_elision elided, integer_core m1)
// -- each computes the SAME fp32 result `*s` as ggml's OWN kernel BIT-FOR-BIT
// (memcmp of the float bits) over random Q4_0 x Q8_0 block arrays, including n
// values that exercise the multi-block MAIN loop and a NONZERO scalar TAIL in the
// SAME accumulation (n=96 nb3 -> mb2 tail 1; n=160 nb5 / 192 nb6 / 224 nb7 ->
// mb4 elided-main + robust-tail mixed). A negative control must FAIL, proving the
// check discriminates.
//
// The three kernels under test are the UNMODIFIED, compiler-emitted C
// (tcrv_emitted_mb{1,2,4}_*.cpp), each #included into its own namespace so the
// shared symbol name resolves to a distinct function. At VLEN=128 the elided
// shape is correct (vsetvl_e8m1(16)=16); the gate runs at VLEN=128 (the board).
//
// REFERENCE: the REAL ggml RVV kernel + _generic, transcribed as in
// inc2a_validate.cpp. Whole TU built with the requested -ffp-contract mode.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

static const int QK = 32;
static const int Q4_STRIDE = 18;
static const int Q8_STRIDE = 34;

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- the three compiler-emitted kernels, each in its own namespace ----------
// Every emitted file declares the SAME extern "C" symbol; wrapping each in a
// namespace gives three distinct callables. The body is byte-for-byte the
// compiler output.
namespace k_mb1 {
#include "body_mb1_robust.inc"
}
namespace k_mb2 {
#include "body_mb2_robust.inc"
}
namespace k_mb4 {
#include "body_mb4_elided.inc"
}

typedef void (*kern_t)(size_t, float *, size_t, const uint8_t *, size_t,
                       const uint8_t *, size_t, int32_t);

// ---- ggml's REAL RVV kernel (transcribed from quants.c:222-271) --------------
static float ggml_real_rvv_s(int n, const uint8_t *vx, const uint8_t *vy) {
  const int qk = QK;
  const int nb = n / qk;
  float sumf = 0;
  size_t vl = qk / 2;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);
    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    uint16_t dx, dy;
    std::memcpy(&dx, xb, 2);
    std::memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  return sumf;
}

// ---- ggml's _generic reference (transcribed from quants.c:174-207) -----------
static float ggml_generic_s(int n, const uint8_t *vx, const uint8_t *vy) {
  const int qk = QK;
  const int nb = n / qk;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    const uint8_t *xqs = xb + 2;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (xqs[j] & 0x0F) - 8;
      const int v1 = (xqs[j] >> 4) - 8;
      sumi0 += (v0 * yqs[j]);
      sumi1 += (v1 * yqs[j + qk / 2]);
    }
    int sumi = sumi0 + sumi1;
    uint16_t dx, dy;
    std::memcpy(&dx, xb, 2);
    std::memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  return sumf;
}

static unsigned g_rng = 0x2468abcdu;
static unsigned next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
static uint16_t random_fp16() {
  uint16_t sign = (next_rand() & 1) << 15;
  uint16_t exp = (uint16_t)(next_rand() % 31);
  uint16_t mant = (uint16_t)(next_rand() & 0x3FF);
  return sign | (exp << 10) | mant;
}
static void fill_blocks(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    uint16_t dx = random_fp16();
    uint16_t dy = random_fp16();
    std::memcpy(xb, &dx, 2);
    std::memcpy(yb, &dy, 2);
    for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(next_rand() & 0xFF);
    for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(next_rand() & 0xFF);
  }
}
static int bits_equal(float a, float b) {
  uint32_t ba, bb;
  std::memcpy(&ba, &a, 4);
  std::memcpy(&bb, &b, 4);
  return ba == bb;
}

struct Shape { const char *name; kern_t fn; };
static int g_failures = 0;
static int g_checked = 0;

static void check_n(const Shape &sh, int n, const char *label) {
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)std::malloc((size_t)nb * Q8_STRIDE);
  fill_blocks(vx, vy, nb);
  float s_ours = 0.0f;
  sh.fn((size_t)n, &s_ours, 0, vx, 0, vy, 0, 1);
  float s_real = ggml_real_rvv_s(n, vx, vy);
  float s_generic = ggml_generic_s(n, vx, vy);
  int real_eq = bits_equal(s_ours, s_real);
  int generic_eq = bits_equal(s_ours, s_generic);
  std::free(vx);
  std::free(vy);
  ++g_checked;
  if (real_eq && generic_eq) {
    if (label)
      printf("PASS [%s n=%d nb=%d] ours=%.9g == ggml_real == ggml_generic\n",
             sh.name, n, nb, (double)s_ours);
    return;
  }
  printf("FAIL [%s n=%d nb=%d] ours=%.9g real=%.9g generic=%.9g "
         "(real_eq=%d generic_eq=%d)\n",
         sh.name, n, nb, (double)s_ours, (double)s_real, (double)s_generic,
         real_eq, generic_eq);
  ++g_failures;
}

int main() {
  Shape shapes[] = {
      {"mb1-robust",
       k_mb1::tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0},
      {"mb2-robust",
       k_mb2::tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0},
      {"mb4-elided",
       k_mb4::tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0},
  };
  // n set: the original block-multiple sizes PLUS sizes whose nb leaves a
  // NONZERO tail with a nonempty main loop (the main->tail seam): n=96 (nb3),
  // 160 (nb5), 192 (nb6), 224 (nb7) -- 160 is the sharp one for mb4 (4 elided
  // main blocks + 1 robust tail block in the same accumulation).
  const int ns[] = {32, 64, 96, 128, 160, 192, 224, 256, 1024, 4096, 8192};
  const int NN = sizeof(ns) / sizeof(ns[0]);
  for (int si = 0; si < 3; ++si) {
    for (int rep = 0; rep < 100; ++rep) {
      for (int i = 0; i < NN; ++i) {
        // Print only the first rep of each n to keep the log readable; every
        // rep is still bit-checked.
        check_n(shapes[si], ns[i], rep == 0 ? "first" : nullptr);
      }
    }
  }

  // ---- negative control: a 1-bit scale flip MUST change *s (non-vacuous) -----
  {
    int n = 160; // nb5: mb4 main(4) + tail(1) -- exercise the seam
    int nb = n / QK;
    uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
    uint8_t *vy = (uint8_t *)std::malloc((size_t)nb * Q8_STRIDE);
    fill_blocks(vx, vy, nb);
    float clean = 0.0f;
    shapes[2].fn((size_t)n, &clean, 0, vx, 0, vy, 0, 1); // mb4-elided
    uint16_t dx;
    std::memcpy(&dx, vx, 2);
    dx ^= 0x8000u;
    std::memcpy(vx, &dx, 2);
    float pert = 0.0f;
    shapes[2].fn((size_t)n, &pert, 0, vx, 0, vy, 0, 1);
    float pert_ref = ggml_real_rvv_s(n, vx, vy);
    std::free(vx);
    std::free(vy);
    int discriminates = !bits_equal(clean, pert);
    int tracks_ref = bits_equal(pert, pert_ref);
    int caught = !bits_equal(clean, pert_ref);
    printf("\nnegative control [mb4-elided n=160]: clean=%.9g perturbed=%.9g "
           "perturbed_ref=%.9g\n",
           (double)clean, (double)pert, (double)pert_ref);
    if (discriminates && tracks_ref && caught) {
      printf("NEGATIVE CONTROL PASS (1-bit flip changed *s; ours tracks ggml on "
             "perturbed data; bitwise check reports MISMATCH on divergent "
             "inputs).\n");
    } else {
      printf("NEGATIVE CONTROL FAIL: discriminates=%d tracks_ref=%d caught=%d\n",
             discriminates, tracks_ref, caught);
      ++g_failures;
    }
  }

  printf("\nINC-5 shape-knob bit-exact: %d checks, %d failures\n", g_checked,
         g_failures);
  printf(g_failures == 0
             ? "RESULT: PASS (all 3 compiler-emitted shapes bitwise equal vs "
               "ggml REAL RVV AND _generic for every n incl. main+tail seams)\n"
             : "RESULT: FAIL\n");
  return g_failures == 0 ? 0 : 1;
}
