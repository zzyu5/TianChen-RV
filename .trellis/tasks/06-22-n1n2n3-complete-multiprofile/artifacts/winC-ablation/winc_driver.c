/* Win-C make-or-break ablation driver for the signed i16 widening dot-reduce
 * family:  out[0] = acc[0] + Sum_i lhs[i]*rhs[i]  (i16*i16 -> i32).
 *
 * THE STRUCTURAL (reduction-structure) ABLATION AT A FIXED LMUL (m1).
 * BOTH timed vector columns are VERBATIM compiler output (tcrv-opt
 * selector/gearbox -> --tcrv-rvv-lower-to-emitc -> mlir-translate --mlir-to-cpp),
 * driven ONLY by the vector_register_budget resource fact crossing the smallest
 * deferred rung's legality threshold (mf2->m1 costs footprint(m1)=1 + reserve 8
 * = 9):
 *
 *   deferred (ON)  : budget 9 -> smallest deferred rung mf2->m1 LEGAL ->
 *                    realizeDeferredWideDotReduceBody: i16mf2 source loads,
 *                    vwmul_vv_i32m1 product, vadd.vv i32m1 DEFERRED accumulate
 *                    into a loop-carried vector, ONE trailing vredsum i32m1->m1.
 *   per-iter (OFF) : budget 8 -> ALL deferred rungs pruned -> selected==nullopt
 *                    -> fall-through (RVVContractionSelectedBodyRealizationOwner
 *                    .cpp:2548) emits the per-iteration WideningDotReduceOp:
 *                    i16mf2 source loads, vwmul_vv_i32m1 product, then a
 *                    vredsum_vs_i32m1 EVERY iteration into the running scalar
 *                    seed (reduction on the loop-carried dependency every iter).
 *
 * BOTH are m1 (i16mf2 source, i32m1 product/result; at VLEN=128 both strip 4
 * elements/iter, identical loads, identical vwmul product). The ONLY difference
 * is the REDUCTION STRUCTURE: deferred-accumulate + one trailing reduce vs a
 * per-iteration vredsum on the loop-carried chain. This isolates the STRUCTURAL
 * axis from the LMUL/budget (Win-A) axis. If the ratio is marginal (~1.0-1.2x)
 * there is NO Win-C (it was Win-A in costume); if ~1.5-3x Win-C is real.
 *
 * This TU (oracle + driver) is compiled -march=rv64gc so the oracle column is
 * genuinely scalar (no vector ISA). The two emitted bodies live in
 * deferred_body.cpp / periter_body.cpp, compiled -march=rv64gcv.
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

/* The two VERBATIM compiler-emitted bodies (renamed symbols _deferred/_periter),
 * defined in deferred_body.cpp / periter_body.cpp, compiled -march=rv64gcv. */
extern void
tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_deferred(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);
extern void
tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_periter(
    const int16_t *, const int16_t *, const int32_t *, int32_t *, size_t);

/* Thin noinline forwarding wrappers so the timed column is the actual compiler
 * output (single tail call, no added vector work). */
__attribute__((noinline)) void
dot_deferred(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
             int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_deferred(
      lhs, rhs, acc, out, n);
}
__attribute__((noinline)) void
dot_periter(const int16_t *lhs, const int16_t *rhs, const int32_t *acc,
            int32_t *out, size_t n) {
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_periter(
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
  dot_deferred(lhs, rhs, acc, out, n);
  int32_t got_deferred = out[0];

  out[0] = 0;
  dot_periter(lhs, rhs, acc, out, n);
  int32_t got_periter = out[0];

  int rc = 0;
  if (got_deferred != oracle) {
    fprintf(stderr, "MISMATCH deferred n=%zu got=%d oracle=%d\n", n,
            got_deferred, oracle);
    rc = 12;
  }
  if (got_periter != oracle) {
    fprintf(stderr, "MISMATCH periter n=%zu got=%d oracle=%d\n", n, got_periter,
            oracle);
    rc = 12;
  }
  if (rc == 0)
    printf("CORRECTNESS n=%zu ok oracle=%d (deferred=%d periter=%d, tol exact)\n",
           n, oracle, got_deferred, got_periter);
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
  dot_fn fns[2] = {dot_deferred, dot_periter};
  const char *names[2] = {"deferred", "periter"};
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

  double deferred = best[0], periter = best[1];
  printf("SUMMARY n=%zu deferred_ns=%.3f periter_ns=%.3f\n", n, deferred,
         periter);
  printf("RATIO n=%zu periter_vs_deferred=%.6f\n", n,
         deferred > 0 ? periter / deferred : 0.0);
  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}

int main(void) {
  printf("CONFIG family=signed-i16-widening-dot-reduce "
         "ablation=STRUCTURAL-deferred-vs-periter-AT-FIXED-LMUL-m1 "
         "deferred=budget9(i16mf2->i32m1,vadd+1trailing-vredsum) "
         "periter=budget8(i16mf2->i32m1,vredsum-every-iter) "
         "both=verbatim-compiler-emitted-SAME-m1-LMUL-only-reduction-structure-differs "
         "warmups=%d repeats=%d iters=%d timing=clock_gettime(MONOTONIC_RAW)\n",
         WARMUPS, REPEATS, ITERS);

  /* Correctness gate: includes odd n=257/1000 to exercise the tail remainder
   * loop (e16mf2 vl=4 at VLEN=128; the 5 timing n are exact multiples, so the
   * odd n is what actually tests the tail). */
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

  printf("PASS winC-structural-ablation sink=%lld\n", (long long)tcrv_sink);
  return 0;
}
