/* STEP-1 ceiling INSURANCE probe (candidate c): UNSIGNED u8 widening
 * product-reduce  sum = acc[0] + Σ lhs[i]*rhs[i]  (u8*u8 -> u32).
 * Same compute intensity as the proven signed i8 byte kernel; reuses the wide
 * deferred-accumulate shape (u8m2 -> u16m4 product -> vwaddu.wv into u32m8).
 * Driver TU compiled -march=rv64gc (genuine scalar); vector variants in
 * ceiling_rvv.c -march=rv64gcv. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUPS 3
#define REPEATS 11
#define ITERS 16

typedef void (*dot_fn)(const uint8_t *, const uint8_t *, const uint32_t *,
                       uint32_t *, size_t);

__attribute__((noinline)) void
u8_scalar(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc,
          uint32_t *out, size_t n) {
  uint32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (uint32_t)lhs[i] * (uint32_t)rhs[i];
  out[0] = sum;
}

void u8_per_iter_reduce(const uint8_t *, const uint8_t *, const uint32_t *,
                        uint32_t *, size_t);
void u8_narrow_deferred(const uint8_t *, const uint8_t *, const uint32_t *,
                        uint32_t *, size_t);
void u8_wide_deferred(const uint8_t *, const uint8_t *, const uint32_t *,
                      uint32_t *, size_t);

static volatile uint64_t tcrv_sink = 0;

static unsigned long long now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) { perror("clock_gettime"); exit(97); }
  return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}

#define N_VARIANTS 4
static const char *VARIANT_NAMES[N_VARIANTS] = {
    "genuine-scalar", "per-iter-reduce", "narrow-deferred", "wide-deferred"};

static int run_case(size_t n) {
  const size_t alloc_n = n == 0 ? 1 : n;
  uint8_t *lhs = (uint8_t *)malloc(alloc_n);
  uint8_t *rhs = (uint8_t *)malloc(alloc_n);
  uint32_t *acc = (uint32_t *)malloc(sizeof(uint32_t) * 4);
  uint32_t *out = (uint32_t *)malloc(sizeof(uint32_t) * 16);
  if (!lhs || !rhs || !acc || !out) { fprintf(stderr, "alloc failed n=%zu\n", n); return 11; }
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (uint8_t)((i % 251) + 1);
    rhs[i] = (uint8_t)(((i * 7 + 3) % 253) + 1);
  }
  acc[0] = (uint32_t)(13 + n);
  acc[1] = acc[2] = acc[3] = 0;

  dot_fn fns[N_VARIANTS] = {u8_scalar, u8_per_iter_reduce, u8_narrow_deferred, u8_wide_deferred};

  out[0] = 0;
  u8_scalar(lhs, rhs, acc, out, n);
  uint32_t oracle = out[0];
  for (int v = 0; v < N_VARIANTS; ++v) {
    out[0] = 0;
    fns[v](lhs, rhs, acc, out, n);
    if (out[0] != oracle) {
      fprintf(stderr, "MISMATCH variant=%s n=%zu got=%u oracle=%u\n",
              VARIANT_NAMES[v], n, out[0], oracle);
      free(lhs); free(rhs); free(acc); free(out); return 12;
    }
  }
  printf("CORRECTNESS n=%zu ok oracle=%u\n", n, oracle);

  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < N_VARIANTS; ++v) { fns[v](lhs, rhs, acc, out, n); tcrv_sink += out[0]; }

  double best[N_VARIANTS];
  for (int v = 0; v < N_VARIANTS; ++v) best[v] = -1.0;
  for (int r = 0; r < REPEATS; ++r)
    for (int v = 0; v < N_VARIANTS; ++v) {
      unsigned long long start = now_ns();
      for (int it = 0; it < ITERS; ++it) { fns[v](lhs, rhs, acc, out, n); tcrv_sink += out[0]; }
      double per_iter = (double)(now_ns() - start) / (double)ITERS;
      if (best[v] < 0.0 || per_iter < best[v]) best[v] = per_iter;
    }

  for (int v = 0; v < N_VARIANTS; ++v)
    printf("SUMMARY variant=%s n=%zu best_per_iter_ns=%.3f\n", VARIANT_NAMES[v], n, best[v]);
  double scal = best[0], periter = best[1], narrow = best[2], wide = best[3];
  printf("RATIO n=%zu wide_vs_scalar=%.6f wide_vs_periter=%.6f "
         "wide_vs_narrow=%.6f narrow_vs_scalar=%.6f periter_vs_scalar=%.6f\n",
         n, wide > 0 ? scal / wide : 0.0, wide > 0 ? periter / wide : 0.0,
         wide > 0 ? narrow / wide : 0.0, narrow > 0 ? scal / narrow : 0.0,
         periter > 0 ? scal / periter : 0.0);
  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}

int main(void) {
  const size_t counts[] = {257, 256, 1024, 4096, 16384, 65536};
  const size_t nc = sizeof(counts) / sizeof(counts[0]);
  printf("CONFIG family=unsigned-u8-widening-product-reduce "
         "variants=genuine-scalar,per-iter-reduce,narrow-deferred,wide-deferred "
         "warmups=%d repeats=%d iters=%d timing=clock_gettime(MONOTONIC_RAW)\n",
         WARMUPS, REPEATS, ITERS);
  for (size_t i = 0; i < nc; ++i) { int s = run_case(counts[i]); if (s != 0) return s; }
  printf("PASS u8-product-reduce ceiling sink=%llu\n", (unsigned long long)tcrv_sink);
  return 0;
}
