// INC-4 microbenchmark (ssh rvv, N3 perf evidence).
//
// Fair 3-way wall-time comparison of ONE ggml_vec_dot_q4_0_q8_0 call over a long
// n (many blocks), best-of-N, identical data, identical build flags:
//   (1) ggml's REAL hand-written RVV kernel (transcribed intrinsic-for-intrinsic
//       from llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:222-271; i8m1 anchor),
//   (2) our OLD compiler-emitted kernel (integer core mf4 -> i16mf2 -> i32m1,
//       4-chunk inner strip + 4 vwredsum per block),
//   (3) our NEW compiler-emitted kernel (integer core m1 -> i16m2 -> i32m1, ONE
//       strip + ONE vwredsum per block; structurally == ggml).
// (2) and (3) are the UNMODIFIED compiler output (tcrv_emitted_kernel_{mf4,m1}.cpp)
// linked under distinct extern symbols. We FIRST assert all three produce the
// bitwise-identical *s on the bench data (no measuring a wrong kernel), then time.
//
// Timing: warmup, then best-of-N repeats of a tight `iters`-call loop, reporting
// the MIN wall-time per call (best-of-N rejects scheduler noise; min is the
// cleanest lower bound). ns/call + GB/s of weight bytes streamed (n/2 q4 bytes).

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

// ---- the two compiler-emitted kernels, linked under distinct symbols ----------
// Both .cpp files define the SAME extern symbol
// tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0, so they are
// compiled into separately-renamed objects (see the build command in RESULTS.md:
// each is preprocessed with -D to rename the symbol). Here we just declare both.
extern "C" void tcrv_kernel_mf4(size_t n, float *s, size_t bs, const uint8_t *vx,
                                size_t bx, const uint8_t *vy, size_t by,
                                int32_t nrc);
extern "C" void tcrv_kernel_m1(size_t n, float *s, size_t bs, const uint8_t *vx,
                               size_t bx, const uint8_t *vy, size_t by,
                               int32_t nrc);

// ---- ggml's REAL RVV kernel (transcribed from quants.c:222-271) ---------------
static void ggml_real_rvv(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
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
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    uint16_t dx, dy;
    std::memcpy(&dx, xb, 2);
    std::memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  *s = sumf;
}

// kern_t-shaped wrapper for the ggml reference (same ABI as the emitted kernels).
static void ggml_real_rvv_wrap(size_t n, float *s, size_t bs, const uint8_t *vx,
                               size_t bx, const uint8_t *vy, size_t by,
                               int32_t nrc) {
  (void)bs; (void)bx; (void)by; (void)nrc;
  ggml_real_rvv((int)n, s, vx, vy);
}

static unsigned g_rng = 0x13572468u;
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
    uint16_t dx = random_fp16(), dy = random_fp16();
    std::memcpy(xb, &dx, 2);
    std::memcpy(yb, &dy, 2);
    for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(next_rand() & 0xFF);
    for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(next_rand() & 0xFF);
  }
}

static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int bits_equal(float a, float b) {
  uint32_t ba, bb;
  std::memcpy(&ba, &a, 4);
  std::memcpy(&bb, &b, 4);
  return ba == bb;
}

typedef void (*kern_t)(size_t, float *, size_t, const uint8_t *, size_t,
                       const uint8_t *, size_t, int32_t);

// best-of-N min ns/call over `iters` calls per repeat.
static double bench(kern_t k, int n, const uint8_t *vx, const uint8_t *vy,
                    int iters, int repeats, volatile float *sink) {
  // warmup
  float s;
  for (int i = 0; i < 64; ++i) k((size_t)n, &s, 0, vx, 0, vy, 0, 1);
  double best = 1e300;
  for (int r = 0; r < repeats; ++r) {
    double t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
      k((size_t)n, &s, 0, vx, 0, vy, 0, 1);
      *sink += s; // defeat dead-store elimination
    }
    double t1 = now_ns();
    double per = (t1 - t0) / iters;
    if (per < best) best = per;
  }
  return best;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 4096;
  int iters = (argc > 2) ? atoi(argv[2]) : 100000;
  int repeats = (argc > 3) ? atoi(argv[3]) : 15;
  if (n % QK != 0) { printf("n must be %% 32\n"); return 2; }
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)std::malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)std::malloc((size_t)nb * Q8_STRIDE);
  fill_blocks(vx, vy, nb);

  // Correctness gate: all three MUST agree bitwise on the bench data.
  float s_g = 0, s_mf4 = 0, s_m1 = 0;
  ggml_real_rvv(n, &s_g, vx, vy);
  tcrv_kernel_mf4((size_t)n, &s_mf4, 0, vx, 0, vy, 0, 1);
  tcrv_kernel_m1((size_t)n, &s_m1, 0, vx, 0, vy, 0, 1);
  printf("correctness gate (n=%d): ggml=%.9g mf4=%.9g m1=%.9g  [mf4==ggml:%d m1==ggml:%d]\n",
         n, (double)s_g, (double)s_mf4, (double)s_m1,
         bits_equal(s_mf4, s_g), bits_equal(s_m1, s_g));
  if (!bits_equal(s_mf4, s_g) || !bits_equal(s_m1, s_g)) {
    printf("ABORT: a kernel disagrees on the bench data; perf numbers would be meaningless.\n");
    return 1;
  }

  volatile float sink = 0;
  double t_ggml = bench(ggml_real_rvv_wrap, n, vx, vy, iters, repeats, &sink);
  double t_mf4  = bench(tcrv_kernel_mf4,   n, vx, vy, iters, repeats, &sink);
  double t_m1   = bench(tcrv_kernel_m1,    n, vx, vy, iters, repeats, &sink);

  double bytes = (double)nb * 16.0; // q4 weight bytes streamed per call
  auto gbps = [&](double ns) { return bytes / ns; }; // bytes/ns == GB/s
  printf("\nn=%d  nb=%d  iters=%d  best-of-%d  (sink=%g)\n", n, nb, iters, repeats, (double)sink);
  printf("  %-22s %10.2f ns/call   %7.2f GB/s\n", "ggml REAL RVV (i8m1)", t_ggml, gbps(t_ggml));
  printf("  %-22s %10.2f ns/call   %7.2f GB/s   (%.2fx ggml)\n", "ours OLD mf4", t_mf4, gbps(t_mf4), t_mf4 / t_ggml);
  printf("  %-22s %10.2f ns/call   %7.2f GB/s   (%.2fx ggml)\n", "ours NEW m1", t_m1, gbps(t_m1), t_m1 / t_ggml);
  printf("\n  m1 vs mf4 speedup: %.2fx   (m1 is %.0f%% of ggml's time)\n",
         t_mf4 / t_m1, 100.0 * t_m1 / t_ggml);

  std::free(vx);
  std::free(vy);
  return 0;
}
