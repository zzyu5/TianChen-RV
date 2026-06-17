// INC-7 q8_0 x q8_0 RIGOROUS microbench (ssh rvv).
//
// Measures the COMPILER's structured emission for q8_0 shapes against ggml's REAL
// hand-written RVV kernel (ggml_vec_dot_q8_0_q8_0, quants.c:435-481, which is
// ITSELF the m2 single-block-elided shape: i8m2->i16m4->vwredsum, ONE per block).
//
//   ggml          : the real ggml_vec_dot_q8_0_q8_0 (i8m2, 1 vwredsum/block)
//   m2_mb1_elided : our compiler's m2 single-block elided (== ggml's structure)
//   m2_mb1_robust : our m2 single-block, strip-loop kept (VLEN-robust)
//   m2_mb2_robust : zve32x autotuner pick (2 blocks/iter, robust)
//   m2_mb4_elided : full-V AUTOTUNER PICK (4 blocks/iter, each ONE vsetvl_e8m2+
//                   ONE vwredsum, NO strip loop, four folds in order, robust
//                   nb%4 tail) -- the latency-overlap lever ggml does NOT use
//
// Methodology mirrors the q4_0 inc6 probe so ns/call is comparable: taskset -c 3
// pinned, best-of-N min ns/call, warmup discarded, identical random data,
// correctness-gated (each shape bit-exact vs ggml on the bench data before
// timing). Build: -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q8_STRIDE 34

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" {
// FAIRNESS: the ggml baseline is a SEPARATE TU (kern_ggml.cpp) compiled with the
// SAME extern "C" + flags as the shapes under test -- no inline-vs-separate-TU
// asymmetry that would bias the baseline. (The inline ggml_real below is used
// ONLY as the bitwise correctness reference, never timed.)
void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
void kern_m2_mb1_elided(size_t, float *, const uint8_t *, const uint8_t *);
void kern_m2_mb1_robust(size_t, float *, const uint8_t *, const uint8_t *);
void kern_m2_mb2_robust(size_t, float *, const uint8_t *, const uint8_t *);
void kern_m2_mb2_elided(size_t, float *, const uint8_t *, const uint8_t *);
void kern_m2_mb4_elided(size_t, float *, const uint8_t *, const uint8_t *);
void kern_q8_compiler_mb2(size_t, float *, const uint8_t *, const uint8_t *);
}

// ggml's REAL hand-written q8_0 RVV kernel (quants.c:435-481).
static void ggml_real(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / QK;
  float sumf = 0;
  size_t vl = QK;
  for (int ib = 0; ib < nb; ++ib) {
    const int8_t *xq = (const int8_t *)(vx + ib * Q8_STRIDE + 2);
    const int8_t *yq = (const int8_t *)(vy + ib * Q8_STRIDE + 2);
    uint16_t xd, yd;
    memcpy(&xd, vx + ib * Q8_STRIDE, 2);
    memcpy(&yd, vy + ib * Q8_STRIDE, 2);
    vint8m2_t bx = __riscv_vle8_v_i8m2(xq, vl);
    vint8m2_t by = __riscv_vle8_v_i8m2(yq, vl);
    vint16m4_t m = __riscv_vwmul_vv_i16m4(bx, by, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r = __riscv_vwredsum_vs_i16m4_i32m1(m, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += sumi * (fp16_to_fp32(xd) * fp16_to_fp32(yd));
  }
  *s = sumf;
}

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void fillb(uint8_t *b, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *p = b + i * Q8_STRIDE;
    _Float16 d = (_Float16)(((float)(int)(xr() % 2001) - 1000.0f) / 256.0f);
    uint16_t dh;
    memcpy(&dh, &d, 2);
    memcpy(p, &dh, 2);
    for (int j = 0; j < QK; j++)
      p[2 + j] = (uint8_t)(int8_t)(xr() % 256);
  }
}

static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

struct Entry { const char *name; kfn fn; double best; double sum; int cnt; };

static double timed_batch(kfn fn, int n, int iters, const uint8_t *vx,
                          const uint8_t *vy) {
  double t0 = now_ns();
  float s;
  for (int i = 0; i < iters; i++)
    fn(n, &s, vx, vy);
  return (now_ns() - t0) / iters;
}

int main() {
  const int n = 4096;        // 128 blocks/call (a typical row chunk)
  const int nb = n / QK;
  const int iters = 2000;    // calls per timed batch
  const int reps = 200;      // best-of-N + INTERLEAVED rounds (anti-drift)
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fillb(vx, nb);
  fillb(vy, nb);

  Entry kernels[] = {
      {"ggml(real,separate-TU)", kern_ggml, 1e18, 0, 0},
      {"m2_mb1_elided", kern_m2_mb1_elided, 1e18, 0, 0},
      {"m2_mb1_robust", kern_m2_mb1_robust, 1e18, 0, 0},
      {"m2_mb2_robust(zve32x-pick)", kern_m2_mb2_robust, 1e18, 0, 0},
      {"m2_mb2_elided", kern_m2_mb2_elided, 1e18, 0, 0},
      {"m2_mb4_elided(OLD-pick,pre-inc8)", kern_m2_mb4_elided, 1e18, 0, 0},
      {"COMPILER-EMITTED mb2 (NEW inc8 pick)", kern_q8_compiler_mb2, 1e18, 0, 0},
  };
  const int K = sizeof(kernels) / sizeof(kernels[0]);

  // Correctness gate: every shape must be bit-exact vs ggml on the bench data.
  float ref = 0;
  ggml_real(n, &ref, vx, vy);
  for (Entry &e : kernels) {
    float got = 0;
    e.fn(n, &got, vx, vy);
    if (memcmp(&got, &ref, 4) != 0) {
      printf("CORRECTNESS GATE FAILED: %s = %.9g != ggml %.9g\n", e.name, got, ref);
      return 1;
    }
  }
  printf("correctness gate: all shapes bit-exact vs ggml on bench data\n");

  // warmup all kernels (so none is cold-penalised on its first timed batch).
  for (Entry &e : kernels)
    for (int i = 0; i < iters; i++) {
      float s;
      e.fn(n, &s, vx, vy);
    }

  // INTERLEAVED rounds: in each round time EVERY kernel once, so board drift /
  // thermal hits all kernels equally rather than penalising whichever is timed
  // last. Track per-kernel best-of-N (min rejects noise) + mean (spread).
  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < K; k++) {
      double per = timed_batch(kernels[k].fn, n, iters, vx, vy);
      if (per < kernels[k].best)
        kernels[k].best = per;
      kernels[k].sum += per;
      kernels[k].cnt += 1;
    }
  }

  double ggml_ns = kernels[0].best;
  printf("%-40s %10s %10s   %s\n", "shape", "best-ns", "mean-ns", "best/ggml");
  for (Entry &e : kernels) {
    double mean = e.sum / e.cnt;
    double ratio = e.best / ggml_ns;
    printf("%-40s %10.1f %10.1f   %.3fx\n", e.name, e.best, mean, ratio);
  }
  free(vx);
  free(vy);
  return 0;
}
