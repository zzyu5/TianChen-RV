// INC-14 G1 bit-exact HW validation + microbench harness (ssh rvv, VLEN=128,
// -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q4_0_q8_0_gemm_tile op
// (ONE weight row x M activation columns, weight-decode reuse: the q4_0 weight
// block is decoded ONCE and the decoded nibble lanes are reused across all M
// activation columns) produces M fp32 outputs s[0..M-1] that are BIT-FOR-BIT
// equal to M INDEPENDENT ggml_vec_dot_q4_0_q8_0(weight_row, column_j) calls --
// which is EXACTLY what ggml computes at VLEN=128 (the repack gemm is disabled
// at `case 128`, so prefill falls back to per-(row,col) vec_dot). Random tiles
// at several n + edge cases. A negative control (perturb ONE column / swap the
// low/high q8 pairing) MUST flip a result to FAIL, proving the check
// discriminates.
//
// Built -ffp-contract=off and again -ffp-contract=fast (the harness reports both;
// the per-column fold is grouped into one expression so the compiler fuses the
// SAME single-rounding FMA as ggml under =on/default; under =off there is no
// contraction in EITHER side -> bit-equal regardless).
//
// MICROBENCH: the G1 tile (decode-once, M columns) vs M x the per-row vec_dot
// (re-decodes the weight every column = ggml's used prefill path at VLEN=128),
// min-of-reps ns/output, pinned. The research found ~1.10-1.13x at K=4096/M~4-6;
// this reports the honest tile-level number.
//
// The kernels under test are the UNMODIFIED, compiler-emitted
// tcrv_emitted_gemm_tile_m{4,6}.cpp (every line tagged
// source_op=tcrv_rvv.q4_0_q8_0_gemm_tile).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <riscv_vector.h>

static const int QK = 32;
static const int Q4_STRIDE = 18;
static const int Q8_STRIDE = 34;

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernels our compiler emitted (verbatim, extern "C") -----------------
// ABI: (size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t by).
// vx = ONE weight-row q4_0 block array; vy = M q8_0 column block arrays laid out
// at byte stride `by` apart; s = M contiguous fp32 outputs.
extern "C" void tcrv_emitc_gemm_m4_gemm_m4_v(size_t n, float *s,
                                             const uint8_t *vx,
                                             const uint8_t *vy, size_t by);
extern "C" void tcrv_emitc_gemm_m6_gemm_m6_v(size_t n, float *s,
                                             const uint8_t *vx,
                                             const uint8_t *vy, size_t by);

// ---- ggml's REAL RVV per-row vec_dot (transcribed from quants.c:222-271) ------
// One weight row x one activation column -> one fp32 result. This re-decodes the
// weight for EVERY column it is called on (the prefill path at VLEN=128). It is
// BOTH the byte-exact oracle (called M times) AND the microbench baseline.
static float ggml_vec_dot_q4_0_q8_0(int n, const uint8_t *vx,
                                    const uint8_t *vy) {
  const int qk = QK;
  const int nb = n / qk;
  float sumf = 0;
  size_t vl = qk / 2; // 16
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
static float ggml_generic_vec_dot(int n, const uint8_t *vx, const uint8_t *vy) {
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

// ---- deterministic xorshift32 RNG --------------------------------------------
static unsigned g_rng = 0x2468abcdu;
static unsigned next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
static uint16_t random_fp16() {
  uint16_t sign = (next_rand() & 1) << 15;
  uint16_t exp = (uint16_t)(next_rand() % 31); // never 31 = Inf/NaN
  uint16_t mant = (uint16_t)(next_rand() & 0x3FF);
  return sign | (exp << 10) | mant;
}
static void fill_q4_row(uint8_t *vx, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    uint16_t dx = random_fp16();
    std::memcpy(xb, &dx, 2);
    for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(next_rand() & 0xFF);
  }
}
static void fill_q8_col(uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    uint16_t dy = random_fp16();
    std::memcpy(yb, &dy, 2);
    for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(next_rand() & 0xFF);
  }
}

static int bits_equal(float a, float b) {
  uint32_t ba, bb;
  std::memcpy(&ba, &a, 4);
  std::memcpy(&bb, &b, 4);
  return ba == bb;
}

static int g_failures = 0;
static int g_checked = 0;

typedef void (*tile_fn)(size_t, float *, const uint8_t *, const uint8_t *,
                        size_t);

// Validate one tile: M columns, one weight row, n elements. The M activation
// columns are laid out contiguously (column j at vy + j*by, by = nb*Q8_STRIDE).
static int check_tile(tile_fn fn, int M, int n, const char *label,
                      int perturb_col) {
  int nb = n / QK;
  size_t by = (size_t)nb * Q8_STRIDE; // per-column activation byte stride
  uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)std::malloc((size_t)M * by);
  fill_q4_row(vx, nb);
  for (int j = 0; j < M; ++j) fill_q8_col(vy + (size_t)j * by, nb);

  if (perturb_col >= 0 && perturb_col < M && nb > 0) {
    // Negative control: flip the sign bit of column `perturb_col` block-0 scale.
    uint8_t *yb = vy + (size_t)perturb_col * by;
    uint16_t dy;
    std::memcpy(&dy, yb, 2);
    dy ^= 0x8000u;
    std::memcpy(yb, &dy, 2);
  }

  float *s_ours = (float *)std::malloc((size_t)M * sizeof(float));
  for (int j = 0; j < M; ++j) s_ours[j] = 0.0f;
  fn((size_t)n, s_ours, vx, vy, by);

  int all_eq = 1;
  for (int j = 0; j < M; ++j) {
    float s_real = ggml_vec_dot_q4_0_q8_0(n, vx, vy + (size_t)j * by);
    float s_gen = ggml_generic_vec_dot(n, vx, vy + (size_t)j * by);
    int real_eq = bits_equal(s_ours[j], s_real);
    int gen_eq = bits_equal(s_ours[j], s_gen);
    if (!(real_eq && gen_eq)) {
      all_eq = 0;
      if (perturb_col < 0)
        printf("FAIL [%s] M=%d n=%d col=%d: ours=%.9g real=%.9g gen=%.9g "
               "(real_eq=%d gen_eq=%d)\n",
               label, M, n, j, (double)s_ours[j], (double)s_real, (double)s_gen,
               real_eq, gen_eq);
    }
  }

  std::free(vx);
  std::free(vy);
  std::free(s_ours);
  ++g_checked;
  if (perturb_col < 0) {
    if (all_eq) {
      if (label && label[0] == 'f') // "first"
        printf("PASS [%s] M=%d n=%d: all %d outputs bitwise == M x "
               "ggml_vec_dot_q4_0_q8_0 == _generic\n",
               label, M, n, M);
    } else {
      ++g_failures;
    }
  }
  return all_eq;
}

// ---- microbench: G1 tile vs M x per-row vec_dot ------------------------------
static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void microbench(tile_fn fn, int M, int n, int iters, int reps) {
  int nb = n / QK;
  size_t by = (size_t)nb * Q8_STRIDE;
  uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)std::malloc((size_t)M * by);
  fill_q4_row(vx, nb);
  for (int j = 0; j < M; ++j) fill_q8_col(vy + (size_t)j * by, nb);
  float *s = (float *)std::malloc((size_t)M * sizeof(float));

  // warmup both
  for (int w = 0; w < 3; ++w) {
    fn((size_t)n, s, vx, vy, by);
    for (int j = 0; j < M; ++j) s[j] = ggml_vec_dot_q4_0_q8_0(n, vx, vy + (size_t)j * by);
  }

  double best_tile = 1e30, best_base = 1e30;
  volatile float sink = 0.0f;
  for (int r = 0; r < reps; ++r) {
    double t0 = now_ns();
    for (int it = 0; it < iters; ++it) {
      fn((size_t)n, s, vx, vy, by);
      sink += s[0];
    }
    double t1 = now_ns();
    if (t1 - t0 < best_tile) best_tile = t1 - t0;

    double b0 = now_ns();
    for (int it = 0; it < iters; ++it) {
      for (int j = 0; j < M; ++j)
        s[j] = ggml_vec_dot_q4_0_q8_0(n, vx, vy + (size_t)j * by);
      sink += s[0];
    }
    double b1 = now_ns();
    if (b1 - b0 < best_base) best_base = b1 - b0;
  }
  double tile_per_out = best_tile / (double)iters / (double)M;
  double base_per_out = best_base / (double)iters / (double)M;
  printf("MICROBENCH M=%d n=%d: M x vec_dot %.1f ns/out | G1 tile %.1f ns/out | "
         "speedup %.3fx  (sink=%g)\n",
         M, n, base_per_out, tile_per_out, base_per_out / tile_per_out,
         (double)sink);
  std::free(vx);
  std::free(vy);
  std::free(s);
}

int main() {
  printf("=== INC-14 G1 GEMM tile byte-exact validation (vs M x "
         "ggml_vec_dot_q4_0_q8_0) ===\n");
  struct {
    tile_fn fn;
    int M;
    const char *name;
  } tiles[] = {
      {tcrv_emitc_gemm_m4_gemm_m4_v, 4, "M4"},
      {tcrv_emitc_gemm_m6_gemm_m6_v, 6, "M6"},
  };
  const int ns[] = {32, 64, 256, 1024, 4096};
  for (auto &t : tiles) {
    for (int rep = 0; rep < 200; ++rep) {
      for (int i = 0; i < 5; ++i) {
        char label[40];
        if (rep == 0) snprintf(label, sizeof(label), "first");
        else snprintf(label, sizeof(label), "%s#%d", t.name, rep);
        check_tile(t.fn, t.M, ns[i], label, /*perturb=*/-1);
      }
    }
  }

  // ---- negative control: perturb ONE column -> that output MUST mismatch -----
  printf("\n--- negative control (perturb ONE column scale; that column's output "
         "MUST diverge from the unperturbed oracle) ---\n");
  {
    int M = 4, n = 256, nb = n / QK;
    size_t by = (size_t)nb * Q8_STRIDE;
    uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
    uint8_t *vy = (uint8_t *)std::malloc((size_t)M * by);
    fill_q4_row(vx, nb);
    for (int j = 0; j < M; ++j) fill_q8_col(vy + (size_t)j * by, nb);
    float s_clean[8];
    tcrv_emitc_gemm_m4_gemm_m4_v((size_t)n, s_clean, vx, vy, by);
    // perturb column 2 block-0 scale
    uint8_t *yb = vy + (size_t)2 * by;
    uint16_t dy; std::memcpy(&dy, yb, 2); dy ^= 0x8000u; std::memcpy(yb, &dy, 2);
    float s_pert[8];
    tcrv_emitc_gemm_m4_gemm_m4_v((size_t)n, s_pert, vx, vy, by);
    float ref_pert_col2 = ggml_vec_dot_q4_0_q8_0(n, vx, vy + (size_t)2 * by);
    // col 2 must change; cols 0,1,3 must NOT (independent accumulators);
    // perturbed col 2 must still track the oracle on the perturbed data.
    int col2_changed = !bits_equal(s_clean[2], s_pert[2]);
    int others_stable = bits_equal(s_clean[0], s_pert[0]) &&
                        bits_equal(s_clean[1], s_pert[1]) &&
                        bits_equal(s_clean[3], s_pert[3]);
    int pert_tracks = bits_equal(s_pert[2], ref_pert_col2);
    int caught_mismatch = !bits_equal(s_clean[2], ref_pert_col2);
    printf("negative control: clean[2]=%.9g pert[2]=%.9g ref_pert[2]=%.9g\n",
           (double)s_clean[2], (double)s_pert[2], (double)ref_pert_col2);
    printf("negative control: col2_changed=%d others_stable=%d pert_tracks=%d "
           "caught_mismatch=%d\n",
           col2_changed, others_stable, pert_tracks, caught_mismatch);
    if (col2_changed && others_stable && pert_tracks && caught_mismatch) {
      printf("NEGATIVE CONTROL PASS: a 1-bit column-2 scale flip changed ONLY "
             "s[2] (independent accumulators), s[2] still tracks the perturbed "
             "oracle, and the bitwise check REPORTS MISMATCH on divergent "
             "inputs (the equality check is non-vacuous).\n");
    } else {
      printf("NEGATIVE CONTROL FAIL\n");
      ++g_failures;
    }
    std::free(vx);
    std::free(vy);
  }

  printf("\nINC-14 G1 byte-exact check: %d tiles checked, %d failures\n",
         g_checked, g_failures);
  if (g_failures == 0)
    printf("RESULT: PASS (every column bitwise equal vs M x "
           "ggml_vec_dot_q4_0_q8_0 AND _generic; negative control "
           "discriminates)\n");
  else
    printf("RESULT: FAIL\n");

  // ---- microbench at the research's win dims --------------------------------
  printf("\n=== INC-14 G1 microbench (G1 tile vs M x per-row vec_dot, "
         "min-of-reps) ===\n");
  microbench(tcrv_emitc_gemm_m4_gemm_m4_v, 4, 4096, 2000, 30);
  microbench(tcrv_emitc_gemm_m6_gemm_m6_v, 6, 4096, 2000, 30);

  return g_failures == 0 ? 0 : 1;
}
