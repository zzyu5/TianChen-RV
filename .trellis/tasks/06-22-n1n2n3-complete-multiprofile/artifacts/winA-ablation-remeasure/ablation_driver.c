/* winA all-compiler LMUL-width ablation driver for the signed i16 widening
 * dot-reduce family:  out[0] = acc[0] + Sum_i lhs[i]*rhs[i]  (i16*i16 -> i32).
 *
 * This is the ALL-COMPILER ablation. BOTH timed vector columns are VERBATIM
 * compiler output (tcrv-opt gearbox/selector -> --tcrv-rvv-lower-to-emitc ->
 * mlir-translate --mlir-to-cpp), differing ONLY in the gearbox's LMUL-width
 * choice driven by the vector_register_budget resource fact:
 *
 *   wide   : budget 32 (default gearbox) -> i16m4 source, i32m8 product==acc,
 *            vadd.vv i32m8 deferred accumulate, ONE trailing vredsum i32m8->m1.
 *   narrow : budget 9 (sed-injected)      -> i16mf2 source, i32m1 product==acc,
 *            vadd.vv i32m1 deferred accumulate, ONE trailing vredsum i32m1->m1.
 *
 * SAME deferred-accumulate algorithm (widening_product + deferred_accumulate +
 * ONE trailing standalone_reduce), ZERO vwredsum, ZERO per-iteration reduce in
 * either body. Only the LMUL width differs. So wide_vs_narrow isolates the
 * gearbox's LMUL-width decision (N3) -- NOT an algorithm swap, NOT a
 * hand-written baseline. This CONFIRMS the existing ~2-4x with all-compiler
 * provenance; it is not a new number.
 *
 * This TU (oracle + driver) is compiled -march=rv64gc so the oracle column is
 * genuinely scalar (no vector ISA). The two emitted bodies live in
 * wide_body.cpp / narrow_body.cpp, compiled -march=rv64gcv.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUPS 3
#define REPEATS 11
#define ITERS 16

typedef void (*dot_fn)(const int16_t *, const int16_t *, const int32_t *,
                       int32_t *, size_t);

/* genuine scalar oracle (this TU, -march=rv64gc => no vector ISA) */
__attribute__((noinline)) void
dot_scalar(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
           int32_t *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = sum;
}

/* The two VERBATIM compiler-emitted bodies (renamed symbols, _wide / _narrow),
 * defined in wide_body.cpp / narrow_body.cpp, compiled -march=rv64gcv. */
extern void
tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_wide(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);
extern void
tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_narrow(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);

/* Thin noinline forwarding wrappers so the timed column is the actual compiler
 * output (single tail call, no added vector work). */
__attribute__((noinline)) void
dot_wide(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
         int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_wide(
      lhs, rhs, acc, out, n);
}
__attribute__((noinline)) void
dot_narrow(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
           int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_narrow(
      lhs, rhs, acc, out, n);
}

static volatile int64_t tcrv_sink = 0;

static unsigned long long now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    perror("clock_gettime");
    exit(97);
  }
  return (unsigned long long)ts.tv_sec * 1000000000ULL +
         (unsigned long long)ts.tv_nsec;
}

static void fill(int16_t *lhs, int16_t *rhs, int32_t *acc, size_t alloc_n,
                 size_t n) {
  /* signed coverage: positive, negative, and products that overflow i16 */
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int16_t)(((i % 6) < 3) ? -((int)(i % 211) + 37)
                                     : ((int)(i % 197) + 41));
    rhs[i] = (int16_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 173) + 29)
                                                      : ((int)(i % 181) + 53));
  }
  acc[0] = (int32_t)(-137 + (int32_t)n);
  acc[1] = acc[2] = acc[3] = 0;
}

/* correctness-only check (includes the odd n that exercises the tail path) */
static int check_case(size_t n) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * 16);
  if (!lhs || !rhs || !acc || !out) {
    fprintf(stderr, "alloc failed n=%zu\n", n);
    return 11;
  }
  fill(lhs, rhs, acc, alloc_n, n);

  out[0] = 0;
  dot_scalar(lhs, rhs, acc, out, n);
  int32_t oracle = out[0];

  out[0] = 0;
  dot_wide(lhs, rhs, acc, out, n);
  int32_t got_wide = out[0];

  out[0] = 0;
  dot_narrow(lhs, rhs, acc, out, n);
  int32_t got_narrow = out[0];

  int rc = 0;
  if (got_wide != oracle) {
    fprintf(stderr, "MISMATCH wide n=%zu got=%d oracle=%d\n", n, got_wide,
            oracle);
    rc = 12;
  }
  if (got_narrow != oracle) {
    fprintf(stderr, "MISMATCH narrow n=%zu got=%d oracle=%d\n", n, got_narrow,
            oracle);
    rc = 12;
  }
  if (rc == 0)
    printf("CORRECTNESS n=%zu ok oracle=%d (wide=%d narrow=%d, tol exact)\n", n,
           oracle, got_wide, got_narrow);
  free(lhs); free(rhs); free(acc); free(out);
  return rc;
}

static int time_case(size_t n) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * 16);
  if (!lhs || !rhs || !acc || !out) {
    fprintf(stderr, "alloc failed n=%zu\n", n);
    return 11;
  }
  fill(lhs, rhs, acc, alloc_n, n);

  /* re-gate at this exact n (exact integer) before timing */
  out[0] = 0;
  dot_scalar(lhs, rhs, acc, out, n);
  int32_t oracle = out[0];
  dot_fn fns[2] = {dot_wide, dot_narrow};
  const char *names[2] = {"wide", "narrow"};
  for (int v = 0; v < 2; ++v) {
    out[0] = 0;
    fns[v](lhs, rhs, acc, out, n);
    if (out[0] != oracle) {
      fprintf(stderr, "MISMATCH variant=%s n=%zu got=%d oracle=%d\n", names[v],
              n, out[0], oracle);
      free(lhs); free(rhs); free(acc); free(out);
      return 12;
    }
  }

  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < 2; ++v) {
      fns[v](lhs, rhs, acc, out, n);
      tcrv_sink += out[0];
    }

  double best[2] = {-1.0, -1.0};
  for (int r = 0; r < REPEATS; ++r)
    for (int v = 0; v < 2; ++v) {
      unsigned long long start = now_ns();
      for (int it = 0; it < ITERS; ++it) {
        fns[v](lhs, rhs, acc, out, n);
        tcrv_sink += out[0];
      }
      double per_iter = (double)(now_ns() - start) / (double)ITERS;
      if (best[v] < 0.0 || per_iter < best[v]) best[v] = per_iter;
    }

  double wide = best[0], narrow = best[1];
  printf("SUMMARY n=%zu wide_ns=%.3f narrow_ns=%.3f\n", n, wide, narrow);
  printf("RATIO n=%zu wide_vs_narrow=%.6f\n", n,
         wide > 0 ? narrow / wide : 0.0);
  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}

int main(void) {
  printf("CONFIG family=signed-i16-widening-dot-reduce "
         "ablation=all-compiler-wide-vs-narrow-LMUL-width "
         "wide=budget32(i16m4->i32m8) narrow=budget9(i16mf2->i32m1) "
         "both=verbatim-compiler-emitted-same-deferred-accumulate-algorithm "
         "warmups=%d repeats=%d iters=%d timing=clock_gettime(MONOTONIC_RAW)\n",
         WARMUPS, REPEATS, ITERS);

  /* Correctness gate: includes odd n=257 to exercise the tail remainder loop
   * (e16m4 vl=32, e16mf2 vl=4 at VLEN=128; the 5 timing n are exact multiples
   * of both, so the odd n is what actually tests the tail). */
  const size_t check_counts[] = {257, 1000, 256, 1024, 4096, 16384, 65536};
  const size_t cc = sizeof(check_counts) / sizeof(check_counts[0]);
  for (size_t i = 0; i < cc; ++i) {
    int s = check_case(check_counts[i]);
    if (s != 0) return s;
  }

  /* Timing: the 5 specified n. */
  const size_t time_counts[] = {256, 1024, 4096, 16384, 65536};
  const size_t tc = sizeof(time_counts) / sizeof(time_counts[0]);
  for (size_t i = 0; i < tc; ++i) {
    int s = time_case(time_counts[i]);
    if (s != 0) return s;
  }

  printf("PASS winA-all-compiler-ablation sink=%lld\n", (long long)tcrv_sink);
  return 0;
}
