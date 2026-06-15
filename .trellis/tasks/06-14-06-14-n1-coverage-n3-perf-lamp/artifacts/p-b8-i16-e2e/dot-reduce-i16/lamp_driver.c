/* STEP-1 ceiling measurement for the 2nd kernel family: signed i16 widening
 * dot-product reduction  sum = acc[0] + Σ lhs[i]*rhs[i]  (i16*i16 -> i32).
 *
 * This is a HAND-WRITTEN ceiling probe (no compiler change yet). It answers the
 * decisive STEP-1 question: does the wide-LMUL deferred-accumulate variant beat
 * BOTH a genuine scalar AND a competent naive RVV for THIS family?
 *
 * Variants (all linked into ONE binary, timed interleaved on the SAME buffers):
 *   0 genuine-scalar    : compiled in this TU as a plain loop. The whole TU is
 *                         compiled -march=rv64gc (no vector ISA) so this column
 *                         is genuinely scalar (objdump-verified by the runner).
 *   1 per-iter-reduce   : the algorithm the COMPILER EMITS TODAY for this family
 *                         (emitWideningDotReduce): vwmul i16mf2->i32m1 + a
 *                         vredsum into a running seed EVERY iteration. The
 *                         latency-bound default the tune must beat.
 *   2 narrow-deferred   : the "naive done right" at narrow LMUL — vwmul
 *                         i16mf2->i32m1 into a persistent i32m1 vector
 *                         accumulator (vadd.vv), ONE trailing vredsum. Competent
 *                         but under-vectorized (mf2 source on a 128-bit board).
 *   3 wide-deferred(m4) : the TUNED candidate — vwmul i16m4->i32m8 into a
 *                         persistent i32m8 accumulator (vadd.vv), ONE trailing
 *                         vredsum i32m8->i32m1. The resource-aware max-legal-LMUL
 *                         choice (i16m4 source, i32m8 product==acc; no 2nd widen).
 *
 * NOTE on accumulate width: unlike the i8 byte kernel (i8m2->i16m4 product, then
 * a WIDENING vwadd.wv into i32m8), here the i16 product is ALREADY i32 — the
 * widened product type == the accumulator type, so the deferred accumulate is a
 * NON-widening vadd.vv. (Confirms (a) needs a new non-widening accumulate op,
 * not the byte path's widening_accumulate.)
 *
 * The vector variants (1,2,3) live in ceiling_rvv.c compiled -march=rv64gcv.
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

/* genuine scalar (this TU, -march=rv64gc => no vector ISA) */
__attribute__((noinline)) void
dot_scalar(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
           int32_t *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = sum;
}

/* vector variants defined in ceiling_rvv.c (compiled -march=rv64gcv) */
void dot_per_iter_reduce(const int16_t *, const int16_t *, const int32_t *,
                         int32_t *, size_t);
void dot_narrow_deferred(const int16_t *, const int16_t *, const int32_t *,
                         int32_t *, size_t);
void dot_wide_deferred(const int16_t *, const int16_t *, const int32_t *,
                       int32_t *, size_t);

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

#define N_VARIANTS 4
static const char *VARIANT_NAMES[N_VARIANTS] = {
    "genuine-scalar", "per-iter-reduce", "narrow-deferred", "wide-deferred"};

static int run_case(size_t n) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * 16);
  if (!lhs || !rhs || !acc || !out) {
    fprintf(stderr, "alloc failed n=%zu\n", n);
    return 11;
  }
  /* signed coverage: positive, negative, and products that overflow i16 */
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int16_t)(((i % 6) < 3) ? -((int)(i % 211) + 37)
                                     : ((int)(i % 197) + 41));
    rhs[i] = (int16_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 173) + 29)
                                                      : ((int)(i % 181) + 53));
  }
  acc[0] = (int32_t)(-137 + (int32_t)n);
  acc[1] = acc[2] = acc[3] = 0;

  dot_fn fns[N_VARIANTS] = {dot_scalar, dot_per_iter_reduce,
                            dot_narrow_deferred, dot_wide_deferred};

  /* correctness guard: scalar oracle, EXACT (integer), before any timing */
  out[0] = 0;
  dot_scalar(lhs, rhs, acc, out, n);
  int32_t oracle = out[0];
  for (int v = 0; v < N_VARIANTS; ++v) {
    out[0] = 0;
    fns[v](lhs, rhs, acc, out, n);
    if (out[0] != oracle) {
      fprintf(stderr, "MISMATCH variant=%s n=%zu got=%d oracle=%d\n",
              VARIANT_NAMES[v], n, out[0], oracle);
      free(lhs); free(rhs); free(acc); free(out);
      return 12;
    }
  }
  printf("CORRECTNESS n=%zu ok oracle=%d\n", n, oracle);

  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < N_VARIANTS; ++v) {
      fns[v](lhs, rhs, acc, out, n);
      tcrv_sink += out[0];
    }

  double best[N_VARIANTS];
  for (int v = 0; v < N_VARIANTS; ++v) best[v] = -1.0;
  for (int r = 0; r < REPEATS; ++r)
    for (int v = 0; v < N_VARIANTS; ++v) {
      unsigned long long start = now_ns();
      for (int it = 0; it < ITERS; ++it) {
        fns[v](lhs, rhs, acc, out, n);
        tcrv_sink += out[0];
      }
      double per_iter = (double)(now_ns() - start) / (double)ITERS;
      if (best[v] < 0.0 || per_iter < best[v]) best[v] = per_iter;
    }

  for (int v = 0; v < N_VARIANTS; ++v)
    printf("SUMMARY variant=%s n=%zu best_per_iter_ns=%.3f\n", VARIANT_NAMES[v],
           n, best[v]);
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
  printf("CONFIG family=signed-i16-widening-dot-reduce "
         "variants=genuine-scalar,per-iter-reduce,narrow-deferred,wide-deferred "
         "warmups=%d repeats=%d iters=%d timing=clock_gettime(MONOTONIC_RAW)\n",
         WARMUPS, REPEATS, ITERS);
  for (size_t i = 0; i < nc; ++i) {
    int s = run_case(counts[i]);
    if (s != 0) return s;
  }
  printf("PASS dot-reduce-i16 ceiling sink=%lld\n", (long long)tcrv_sink);
  return 0;
}
