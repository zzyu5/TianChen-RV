// INC-25 GEMM G2 byte-exact HW validation + microbench harness (ssh rvv,
// VLEN=128, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q4_0_q8_0_gemm op (the
// FULL Q4_0 x Q8_0 GEMM: NR weight rows x nc activation columns, wrapping G1's
// weight-decode-reuse tile in an outer weight-ROW loop x an inner M-wide
// column-strip loop) produces an NR x nc fp32 output matrix s[row][col] that is
// BIT-FOR-BIT equal to NR*nc INDEPENDENT ggml_vec_dot_q4_0_q8_0(weight_row_row,
// activation_col_col) calls -- which is EXACTLY what ggml computes at VLEN=128
// (the repack gemm is disabled at `case 128`, so prefill falls back to
// per-(row,col) vec_dot). Random matrices at several (NR, nc, K) incl. the
// llama-2-7b dims, AND deliberate tail shapes (nc not a multiple of M). A
// negative control (perturb ONE weight scale) MUST flip the affected row to
// FAIL, proving the check discriminates.
//
// Built -ffp-contract=off and again -ffp-contract=fast (the harness reports
// both; the per-column fold is grouped into one expression so the compiler
// fuses the SAME single-rounding FMA as ggml under =on/default; under =off there
// is no contraction in EITHER side -> bit-equal regardless).
//
// MICROBENCH: the full G2 GEMM (decode-once per (row, block), M columns reuse it)
// vs NR*nc x the per-row vec_dot (re-decodes the weight every output column =
// ggml's used prefill path at VLEN=128), min-of-reps ns/output, pinned. The
// research found ~1.10x blended / ~1.13x tile at K=4096/M~4-6; this reports the
// honest full-GEMM number at the realistic K.
//
// The kernels under test are the UNMODIFIED, compiler-emitted
// tcrv_emitted_gemm_m{4,6}.cpp (every line tagged
// source_op=tcrv_rvv.q4_0_q8_0_gemm).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <vector>
#include <riscv_vector.h>

static const int QK = 32;
static const int Q4_STRIDE = 18; // sizeof(block_q4_0) = 2 (fp16 d) + 16 (qs)
static const int Q8_STRIDE = 34; // sizeof(block_q8_0) = 2 (fp16 d) + 32 (qs)

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}
static inline uint16_t fp32_to_fp16(float f) {
  _Float16 hf = (_Float16)f;
  uint16_t h;
  std::memcpy(&h, &hf, sizeof(uint16_t));
  return h;
}

// ---- The kernels our compiler emitted (verbatim, extern "C") -----------------
// ABI: (size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t by,
//       size_t nr, size_t nc, size_t bx, size_t bs).
// vx = NR weight rows (each a q4_0 block array), at byte stride bx apart.
// vy = nc q8_0 column block arrays, at byte stride by apart.
// s  = NR x nc fp32 outputs, at float stride bs per row.
extern "C" void tcrv_emitc_gemm_m4_gemm_m4(size_t n, float *s,
                                           const uint8_t *vx, const uint8_t *vy,
                                           size_t by, size_t nr, size_t nc,
                                           size_t bx, size_t bs);
extern "C" void tcrv_emitc_gemm_m6_gemm_m6(size_t n, float *s,
                                           const uint8_t *vx, const uint8_t *vy,
                                           size_t by, size_t nr, size_t nc,
                                           size_t bx, size_t bs);

// ---- ggml's REAL RVV per-row vec_dot (transcribed from quants.c:222-271) ------
// One weight row x one activation column -> one fp32 result. This re-decodes the
// weight for EVERY column it is called on (the prefill path at VLEN=128). It is
// BOTH the byte-exact oracle (called NR*nc times) AND the microbench baseline.
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

    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);

    vint32m1_t seed = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(p, seed, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);

    float d_x = fp16_to_fp32(*(const uint16_t *)xb);
    float d_y = fp16_to_fp32(*(const uint16_t *)yb);
    sumf = sumf + ((float)sumi * d_x) * d_y;
  }
  return sumf;
}

// ---- random q4_0 / q8_0 block generators ------------------------------------
static uint64_t rng_state = 0x123456789abcdef0ULL;
static uint32_t xrand() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}

static void fill_q4_0_row(uint8_t *row, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *b = row + (size_t)ib * Q4_STRIDE;
    float d = ((float)(int)(xrand() % 2000) - 1000.0f) / 4096.0f;
    *(uint16_t *)b = fp32_to_fp16(d);
    for (int j = 0; j < 16; ++j)
      b[2 + j] = (uint8_t)(xrand() & 0xFF);
  }
}
static void fill_q8_0_col(uint8_t *col, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *b = col + (size_t)ib * Q8_STRIDE;
    float d = ((float)(int)(xrand() % 2000) - 1000.0f) / 8192.0f;
    *(uint16_t *)b = fp32_to_fp16(d);
    for (int j = 0; j < 32; ++j)
      b[2 + j] = (int8_t)(xrand() & 0xFF);
  }
}

// ---- one validation case: NR rows x nc cols, contraction K = n --------------
typedef void (*gemm_fn)(size_t, float *, const uint8_t *, const uint8_t *,
                        size_t, size_t, size_t, size_t, size_t);

static int validate_case(gemm_fn fn, const char *tag, int n, int nr, int nc,
                         int perturb_row /* -1 = none */) {
  int nb = n / QK;
  size_t bx = (size_t)nb * Q4_STRIDE; // q4_0 weight-row byte stride
  size_t by = (size_t)nb * Q8_STRIDE; // q8_0 column byte stride
  size_t bs = (size_t)nc;             // dense output row stride (floats)

  std::vector<uint8_t> W((size_t)nr * bx);
  std::vector<uint8_t> A((size_t)nc * by);
  for (int r = 0; r < nr; ++r)
    fill_q4_0_row(W.data() + (size_t)r * bx, nb);
  for (int c = 0; c < nc; ++c)
    fill_q8_0_col(A.data() + (size_t)c * by, nb);

  // Oracle: NR*nc independent per-(row,col) vec_dot.
  std::vector<float> ref((size_t)nr * nc);
  for (int r = 0; r < nr; ++r)
    for (int c = 0; c < nc; ++c)
      ref[(size_t)r * nc + c] =
          ggml_vec_dot_q4_0_q8_0(n, W.data() + (size_t)r * bx,
                                 A.data() + (size_t)c * by);

  // Optional negative control: flip ONE bit in row `perturb_row`'s first scale
  // AFTER computing the ref -> the kernel must then DIVERGE on that row only.
  if (perturb_row >= 0 && perturb_row < nr) {
    uint8_t *b = W.data() + (size_t)perturb_row * bx;
    *(uint16_t *)b ^= 0x0001; // perturb the fp16 scale's low bit
  }

  std::vector<float> got((size_t)nr * nc, -123456.0f);
  fn((size_t)n, got.data(), W.data(), A.data(), by, (size_t)nr, (size_t)nc, bx,
     bs);

  int mism = 0, perturbed_row_diff = 0;
  for (int r = 0; r < nr; ++r) {
    for (int c = 0; c < nc; ++c) {
      uint32_t a, b;
      float gv = got[(size_t)r * nc + c], rv = ref[(size_t)r * nc + c];
      std::memcpy(&a, &gv, 4);
      std::memcpy(&b, &rv, 4);
      if (a != b) {
        if (r == perturb_row)
          perturbed_row_diff++;
        else {
          if (mism < 4)
            printf("  [%s] MISMATCH s[%d][%d] got=%.9g ref=%.9g (bits %08x vs "
                   "%08x)\n",
                   tag, r, c, gv, rv, a, b);
          mism++;
        }
      }
    }
  }

  if (perturb_row >= 0) {
    // Negative control: the perturbed row MUST differ, all OTHER rows must match.
    printf("  [%s] negative-control n=%d nr=%d nc=%d perturb_row=%d: "
           "perturbed-row-diffs=%d other-row-mismatches=%d -> %s\n",
           tag, n, nr, nc, perturb_row, perturbed_row_diff, mism,
           (perturbed_row_diff > 0 && mism == 0) ? "OK(discriminates)"
                                                 : "BAD");
    return (perturbed_row_diff > 0 && mism == 0) ? 0 : 1;
  }
  printf("  [%s] n=%d nr=%d nc=%d : %d mismatches -> %s\n", tag, n, nr, nc, mism,
         mism == 0 ? "PASS" : "FAIL");
  return mism;
}

// ---- microbench: full GEMM vs NR*nc per-row vec_dot --------------------------
static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void microbench(gemm_fn fn, const char *tag, int n, int nr, int nc,
                       int reps) {
  int nb = n / QK;
  size_t bx = (size_t)nb * Q4_STRIDE, by = (size_t)nb * Q8_STRIDE,
         bs = (size_t)nc;
  std::vector<uint8_t> W((size_t)nr * bx);
  std::vector<uint8_t> A((size_t)nc * by);
  for (int r = 0; r < nr; ++r)
    fill_q4_0_row(W.data() + (size_t)r * bx, nb);
  for (int c = 0; c < nc; ++c)
    fill_q8_0_col(A.data() + (size_t)c * by, nb);
  std::vector<float> out((size_t)nr * nc);

  // Warm.
  fn((size_t)n, out.data(), W.data(), A.data(), by, nr, nc, bx, bs);
  for (int r = 0; r < nr; ++r)
    for (int c = 0; c < nc; ++c)
      out[(size_t)r * nc + c] =
          ggml_vec_dot_q4_0_q8_0(n, W.data() + (size_t)r * bx,
                                 A.data() + (size_t)c * by);

  double best_gemm = 1e30, best_base = 1e30;
  volatile float sink = 0;
  for (int rep = 0; rep < reps; ++rep) {
    double t0 = now_ns();
    fn((size_t)n, out.data(), W.data(), A.data(), by, nr, nc, bx, bs);
    double t1 = now_ns();
    if (t1 - t0 < best_gemm)
      best_gemm = t1 - t0;
    sink += out[0];
  }
  for (int rep = 0; rep < reps; ++rep) {
    double t0 = now_ns();
    for (int r = 0; r < nr; ++r)
      for (int c = 0; c < nc; ++c)
        out[(size_t)r * nc + c] =
            ggml_vec_dot_q4_0_q8_0(n, W.data() + (size_t)r * bx,
                                   A.data() + (size_t)c * by);
    double t1 = now_ns();
    if (t1 - t0 < best_base)
      best_base = t1 - t0;
    sink += out[0];
  }
  double outputs = (double)nr * nc;
  printf("  [%s] BENCH n=%d nr=%d nc=%d : full-GEMM %.1f ns/out  | NR*nc "
         "vec_dot %.1f ns/out  | speedup %.3fx  (sink=%.3g)\n",
         tag, n, nr, nc, best_gemm / outputs, best_base / outputs,
         best_base / best_gemm, (double)sink);
}

int main() {
  int fail = 0;
  printf("=== INC-25 GEMM G2 byte-exact validation (M=4 kernel) ===\n");
  // Block-multiple and tail nc (nc % 4 != 0 exercises the strip tail clamp).
  int ns[] = {32, 64, 256, 1024, 4096};
  int shapes[][2] = {{8, 4}, {8, 6}, {16, 7}, {3, 5}, {5, 1}, {7, 13}};
  for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); ++si)
    for (int ni = 0; ni < (int)(sizeof(ns) / sizeof(ns[0])); ++ni)
      fail += validate_case(tcrv_emitc_gemm_m4_gemm_m4, "M4", ns[ni],
                            shapes[si][0], shapes[si][1], -1);

  printf("=== INC-25 GEMM G2 byte-exact validation (M=6 kernel) ===\n");
  for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); ++si)
    for (int ni = 0; ni < (int)(sizeof(ns) / sizeof(ns[0])); ++ni)
      fail += validate_case(tcrv_emitc_gemm_m6_gemm_m6, "M6", ns[ni],
                            shapes[si][0], shapes[si][1], -1);

  printf("=== negative controls (must discriminate) ===\n");
  fail += validate_case(tcrv_emitc_gemm_m4_gemm_m4, "M4", 256, 8, 6, 3);
  fail += validate_case(tcrv_emitc_gemm_m6_gemm_m6, "M6", 256, 8, 6, 5);

  printf("=== microbench (full GEMM vs NR*nc per-row vec_dot) ===\n");
  // Realistic llama-2-7b contraction dims; nr=8 weight-row tile.
  microbench(tcrv_emitc_gemm_m4_gemm_m4, "M4", 4096, 8, 4, 4000);
  microbench(tcrv_emitc_gemm_m4_gemm_m4, "M4", 4096, 8, 8, 4000);
  microbench(tcrv_emitc_gemm_m6_gemm_m6, "M6", 4096, 8, 6, 4000);
  microbench(tcrv_emitc_gemm_m6_gemm_m6, "M6", 4096, 8, 12, 3000);
  microbench(tcrv_emitc_gemm_m4_gemm_m4, "M4", 11008, 8, 4, 1500);
  microbench(tcrv_emitc_gemm_m6_gemm_m6, "M6", 11008, 8, 6, 1500);

  printf("\n=== RESULT: %s (total non-control failures + bad controls = %d) "
         "===\n",
         fail == 0 ? "ALL PASS" : "FAILURES", fail);
  return fail == 0 ? 0 : 1;
}
