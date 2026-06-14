#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUPS 3
#define REPEATS 11
#define ITERS 16
#define N_VARIANTS 15

typedef void (*kernel_fn)(const int8_t *, const int8_t *, const int32_t *,
                          float, float *, size_t);

void ref_scalar_byte(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void naive_anchor_byte(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f4_a1(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f4_a2(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f4_a4(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f4_a8(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f2_a1(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f2_a2(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f2_a4(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_f2_a8(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_m1_a1(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_m1_a2(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_m1_a4(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_m2_a1(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void v_m2_a2(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);

static const char *VARIANT_NAMES[N_VARIANTS] = { "genuine-scalar", "naive-mf4-anchor", "L=mf4_A=1", "L=mf4_A=2", "L=mf4_A=4", "L=mf4_A=8", "L=mf2_A=1", "L=mf2_A=2", "L=mf2_A=4", "L=mf2_A=8", "L=m1_A=1", "L=m1_A=2", "L=m1_A=4", "L=m2_A=1", "L=m2_A=2" };

static volatile double tcrv_sink = 0.0;

static inline uint64_t rdcycle_now(void) {
  uint64_t v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}

static unsigned long long now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) { perror("clock_gettime"); exit(97); }
  return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

static int run_case(size_t n, float scale) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int8_t *lhs = (int8_t *)malloc(alloc_n);
  int8_t *rhs = (int8_t *)malloc(alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  float *out = (float *)malloc(sizeof(float) * 16);
  if (!lhs || !rhs || !acc || !out) { fprintf(stderr, "alloc failed n=%zu\n", n); return 11; }

  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int8_t)(((i % 6) < 3) ? -((int)(i % 53) + 21) : ((int)(i % 47) + 18));
    rhs[i] = (int8_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 43) + 17)
                                                     : ((int)(i % 41) + 23));
  }

  acc[0] = (int32_t)(-137 + (int32_t)n);
  acc[1] = acc[2] = acc[3] = 0;

  kernel_fn fns[N_VARIANTS] = {
    ref_scalar_byte,
    naive_anchor_byte,
    v_f4_a1,
    v_f4_a2,
    v_f4_a4,
    v_f4_a8,
    v_f2_a1,
    v_f2_a2,
    v_f2_a4,
    v_f2_a8,
    v_m1_a1,
    v_m1_a2,
    v_m1_a4,
    v_m2_a1,
    v_m2_a2
  };

  /* correctness guard: scalar oracle, every variant, tol 1e-05, before timing. */
  const float tol = 1e-05f;
  out[0] = 0.0f;
  fns[0](lhs, rhs, acc, scale, out, n);
  float oracle = out[0];
  for (int v = 0; v < N_VARIANTS; ++v) {
    out[0] = 0.0f;
    fns[v](lhs, rhs, acc, scale, out, n);
    float d = out[0] - oracle; if (d < 0) d = -d;
    if (d > tol) {
      fprintf(stderr, "MISMATCH variant=%s n=%zu got=%.9g oracle=%.9g delta=%.9g\n",
              VARIANT_NAMES[v], n, out[0], oracle, d);
      free(lhs); free(rhs); free(acc); free(out);
      return 12;
    }
  }
  printf("CORRECTNESS n=%zu ok oracle=%.9g\n", n, oracle);

  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < N_VARIANTS; ++v) { fns[v](lhs, rhs, acc, scale, out, n); tcrv_sink += (double)out[0]; }

  double best_ns[N_VARIANTS];
  unsigned long long best_cyc[N_VARIANTS];
  for (int v = 0; v < N_VARIANTS; ++v) { best_ns[v] = -1.0; best_cyc[v] = 0; }

  for (int r = 0; r < REPEATS; ++r) {
    for (int v = 0; v < N_VARIANTS; ++v) {
      uint64_t c0 = rdcycle_now();
      unsigned long long t0 = now_ns();
      for (int it = 0; it < ITERS; ++it) { fns[v](lhs, rhs, acc, scale, out, n); tcrv_sink += (double)out[0]; }
      unsigned long long dt = now_ns() - t0;
      uint64_t dc = rdcycle_now() - c0;
      double per_ns = (double)dt / (double)ITERS;
      unsigned long long per_cyc = dc / (unsigned long long)ITERS;
      if (best_ns[v] < 0.0 || per_ns < best_ns[v]) best_ns[v] = per_ns;
      if (best_cyc[v] == 0 || per_cyc < best_cyc[v]) best_cyc[v] = per_cyc;
    }
  }

  for (int v = 0; v < N_VARIANTS; ++v)
    printf("SUMMARY variant=%s n=%zu best_ns=%.3f best_cyc=%llu\n",
           VARIANT_NAMES[v], n, best_ns[v], best_cyc[v]);

  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}

int main(void) {
  const size_t counts[] = { 257, 256, 1024, 4096, 16384, 65536 };
  const float scale = 0.0125f;
  const size_t nc = sizeof(counts) / sizeof(counts[0]);
  printf("CONFIG nvar=%d counts=257,256,1024,4096,16384,65536 warmups=%d repeats=%d iters=%d "
         "timing=rdcycle+clock_gettime\n", N_VARIANTS, WARMUPS, REPEATS, ITERS);
  for (size_t i = 0; i < nc; ++i) { int s = run_case(counts[i], scale); if (s != 0) return s; }
  printf("PASS accumulator-sweep counts=257,256,1024,4096,16384,65536 sink=%.9g\n", tcrv_sink);
  return 0;
}
