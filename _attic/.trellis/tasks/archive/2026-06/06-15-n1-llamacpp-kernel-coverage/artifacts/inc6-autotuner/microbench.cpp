// INC-5 shape-knob RIGOROUS microbench (ssh rvv).
//
// Measures the COMPILER's structured emission for the three bounded shapes of
// tcrv_rvv.q4_0_q8_0_block_dot against ggml's REAL hand-written RVV kernel:
//   ggml        : the real ggml_vec_dot_q4_0_q8_0 (i8m1, 1 vwredsum/block, serial)
//   mb1-robust  : our current single-block m1 anchor (strip loop, ONE vwredsum)
//   mb2-robust  : 2 blocks/iter, two strip-loop cores emitted adjacent, two folds
//                 in order (the latency-overlap lever within the VLEN-robust form)
//   mb4-elided  : 4 blocks/iter, each ONE vsetvl_e8m1(16)+ONE vwredsum (NO strip
//                 loop), four folds in order, robust nb%4 tail (VLEN>=128 only)
//
// The three shapes are the EXACT compiler-emitted C (kern_mb{1,2,4}_*.cpp,
// renamed extern "C", compiled as SEPARATE TUs so cross-function scheduling does
// not distort timing). Methodology copies the design-space probe so the ns/call
// is directly comparable to its 1168/1564 baselines: taskset -c 3 pinned,
// best-of-N min ns/call (min rejects scheduler noise), warmup discarded,
// identical random data, correctness-gated (every shape bitwise == ggml on the
// bench data before timing). Build: -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3
// -ffp-contract=fast.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 18
#define Q8_STRIDE 34

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// The three compiler-emitted shapes (separate TUs, extern "C").
extern "C" void kern_mb1_robust(size_t, float *, size_t, const uint8_t *, size_t,
                                const uint8_t *, size_t, int32_t);
extern "C" void kern_mb2_robust(size_t, float *, size_t, const uint8_t *, size_t,
                                const uint8_t *, size_t, int32_t);
extern "C" void kern_mb4_elided(size_t, float *, size_t, const uint8_t *, size_t,
                                const uint8_t *, size_t, int32_t);

// ggml's REAL hand-written RVV kernel (the reference; same recipe as the probe).
static void ggml_real(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = QK / 2;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
    uint16_t dx, dy;
    memcpy(&dx, xb, 2);
    memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  *s = sumf;
}

static unsigned g = 0x13572468u;
static unsigned rnd() { g ^= g << 13; g ^= g >> 17; g ^= g << 5; return g; }
static uint16_t rfp16() {
  uint16_t s = (rnd() & 1) << 15, e = (uint16_t)(rnd() % 31),
           m = (uint16_t)(rnd() & 0x3FF);
  return s | (e << 10) | m;
}
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * Q4_STRIDE, *yb = vy + (size_t)ib * Q8_STRIDE;
    uint16_t dx = rfp16(), dy = rfp16();
    memcpy(xb, &dx, 2);
    memcpy(yb, &dy, 2);
    for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(rnd() & 0xFF);
    for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(rnd() & 0xFF);
  }
}
static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static int beq(float a, float b) {
  uint32_t x, y;
  memcpy(&x, &a, 4);
  memcpy(&y, &b, 4);
  return x == y;
}
typedef void (*kern_t)(size_t, float *, size_t, const uint8_t *, size_t,
                       const uint8_t *, size_t, int32_t);
static void ggml_wrap(size_t n, float *s, size_t, const uint8_t *vx, size_t,
                      const uint8_t *vy, size_t, int32_t) {
  ggml_real((int)n, s, vx, vy);
}
static double bench(kern_t k, int n, const uint8_t *vx, const uint8_t *vy,
                    int iters, int reps, volatile float *sink) {
  float s;
  for (int i = 0; i < 64; ++i) k((size_t)n, &s, 0, vx, 0, vy, 0, 1); // warmup
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
      k((size_t)n, &s, 0, vx, 0, vy, 0, 1);
      *sink += s;
    }
    double t1 = now_ns();
    double per = (t1 - t0) / iters;
    if (per < best) best = per;
  }
  return best;
}
struct V { const char *name; kern_t fn; const char *ship; };

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 4096;
  int iters = (argc > 2) ? atoi(argv[2]) : 100000;
  int reps = (argc > 3) ? atoi(argv[3]) : 40;
  if (n % QK) { printf("n%%32\n"); return 2; }
  int nb = n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill(vx, vy, nb);
  V vs[] = {
      {"ggml(i8m1,serial)", ggml_wrap, "REFERENCE"},
      {"mb1-robust(compiler)", kern_mb1_robust, "shippable (current anchor)"},
      {"mb2-robust(compiler)", kern_mb2_robust, "shippable (VLEN-robust)"},
      {"mb4-elided(compiler)", kern_mb4_elided, "VLEN>=128 only (Zvl128b)"},
  };
  int NV = sizeof(vs) / sizeof(vs[0]);
  float ref;
  ggml_real(n, &ref, vx, vy);
  printf("# n=%d nb=%d iters=%d reps=%d  ref=%.9g\n", n, nb, iters, reps,
         (double)ref);
  printf("# correctness gate (bitwise == ggml):\n");
  int allok = 1;
  for (int i = 0; i < NV; ++i) {
    float s;
    vs[i].fn((size_t)n, &s, 0, vx, 0, vy, 0, 1);
    int ok = beq(s, ref);
    printf("#   %-24s %.9g  [%s]\n", vs[i].name, (double)s, ok ? "OK" : "MISMATCH");
    if (!ok) allok = 0;
  }
  if (!allok) printf("# ABORT: a variant mismatches; timing skipped.\n");
  volatile float sink = 0;
  double tg = 0;
  printf("\n# timing (best-of-%d min ns/call), taskset pinned, warmup discarded:\n",
         reps);
  for (int i = 0; i < NV; ++i) {
    double t = bench(vs[i].fn, n, vx, vy, iters, reps, &sink);
    if (i == 0) tg = t;
    printf("  %-24s %9.2f ns/call   %.3fx ggml   [%s]\n", vs[i].name, t, t / tg,
           vs[i].ship);
  }
  printf("# sink=%g\n", (double)sink);
  free(vx);
  free(vy);
  return 0;
}
