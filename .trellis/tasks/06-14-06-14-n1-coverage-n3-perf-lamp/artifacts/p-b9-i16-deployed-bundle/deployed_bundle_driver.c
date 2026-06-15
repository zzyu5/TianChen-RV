/* P-B9 deployed-bundle ssh-rvv correctness driver (I8).
 *
 * Compiles + links the GENERATED deferred-wide i16 dot-reduce kernel OBJECT (.o)
 * and calls it through the GENERATED callable-C HEADER (.h) -- i.e. exercises the
 * deployable target-artifact BUNDLE end to end, not the runnable C. Validates the
 * bundle's numeric output against a genuine scalar oracle (== scalar, exact for
 * integers) across a count sweep including the prime-tail partial strip (n=257)
 * and large n.
 *
 * The kernel computes out[0] = acc[0] + sum_i (int32)lhs[i] * (int32)rhs[i], with
 * lhs/rhs i16 sources, accumulated wide in i32 (deferred i32m8 vadd.vv) and reduced
 * once. The deployed bundle path -- not the runnable C -- runs numerically correct.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The GENERATED callable-C header carries the kernel prototype. */
#include "i16_dot_reduce_kernel.h"

static int16_t lhs_pat(size_t i) {
  return (int16_t)(((i % 4) < 2) ? -((int)(i % 277) + 31) : ((int)(i % 263) + 19));
}
static int16_t rhs_pat(size_t i) {
  return (int16_t)(((i % 5) == 0) ? -((int)(i % 211) + 23) : ((int)(i % 197) + 13));
}

/* Genuine scalar oracle: acc[0] + sum_i lhs[i]*rhs[i] in i32. */
static int32_t scalar_oracle(const int16_t *lhs, const int16_t *rhs,
                             int32_t acc0, size_t n) {
  int32_t dot = acc0;
  for (size_t i = 0; i < n; ++i)
    dot += (int32_t)lhs[i] * (int32_t)rhs[i];
  return dot;
}

static int run_case(size_t n, int32_t acc0) {
  int16_t *lhs = (int16_t *)malloc((n ? n : 1) * sizeof(int16_t));
  int16_t *rhs = (int16_t *)malloc((n ? n : 1) * sizeof(int16_t));
  int32_t acc = acc0;
  int32_t out = 1234567; /* poison */
  for (size_t i = 0; i < n; ++i) {
    lhs[i] = lhs_pat(i);
    rhs[i] = rhs_pat(i);
  }
  /* Call through the GENERATED header prototype into the GENERATED .o. */
  tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(
      lhs, rhs, &acc, &out, n);
  int32_t ref = scalar_oracle(lhs, rhs, acc0, n);
  int32_t err = out - ref;
  if (err < 0)
    err = -err;
  printf("n=%-6zu acc0=%-4d bundle_out=%-12d scalar_ref=%-12d abs_err=%d %s\n",
         n, acc0, out, ref, err, err == 0 ? "PASS" : "FAIL");
  free(lhs);
  free(rhs);
  return err == 0 ? 0 : 1;
}

int main(void) {
  size_t counts[] = {0, 1, 16, 17, 256, 257, 1024, 4096, 16384, 65536};
  int32_t accs[] = {0, 7, -19};
  int fails = 0;
  for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    for (size_t a = 0; a < sizeof(accs) / sizeof(accs[0]); ++a)
      fails += run_case(counts[c], accs[a]);
  printf(fails == 0 ? "DEPLOYED_BUNDLE_ALL_PASS\n" : "DEPLOYED_BUNDLE_FAIL\n");
  return fails == 0 ? 0 : 1;
}
