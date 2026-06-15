/* P-B6 deployed-bundle ssh-rvv correctness driver (I8).
 *
 * Compiles + links the GENERATED deferred-wide byte kernel OBJECT (.o) and calls
 * it through the GENERATED callable-C HEADER (.h) -- i.e. exercises the deployable
 * target-artifact BUNDLE end to end, not the runnable C. Validates the bundle's
 * numeric output against a genuine scalar oracle (== scalar, tol 1e-05) across a
 * count sweep including the prime-tail partial strip (n=257) and large n.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* The GENERATED callable-C header carries the kernel prototype. */
#include "deferred_wide_byte_kernel.h"

static int8_t lhs_pat(size_t i) {
  return (int8_t)(((i % 4) < 2) ? -((int)(i % 47) + 12) : ((int)(i % 43) + 14));
}
static int8_t rhs_pat(size_t i) {
  return (int8_t)(((i % 5) == 0) ? -((int)(i % 31) + 15) : ((int)(i % 29) + 17));
}

/* Genuine scalar oracle: acc[0] + sum_i lhs[i]*rhs[i], dequantized by scale. */
static float scalar_oracle(const int8_t *lhs, const int8_t *rhs, int32_t acc0,
                           float scale, size_t n) {
  int32_t dot = acc0;
  for (size_t i = 0; i < n; ++i)
    dot += (int32_t)lhs[i] * (int32_t)rhs[i];
  return ((float)dot) * scale;
}

static int run_case(size_t n, int32_t acc0, float scale) {
  int8_t *lhs = (int8_t *)malloc(n ? n : 1);
  int8_t *rhs = (int8_t *)malloc(n ? n : 1);
  int32_t acc = acc0;
  float out = 1234.5f; /* poison */
  for (size_t i = 0; i < n; ++i) {
    lhs[i] = lhs_pat(i);
    rhs[i] = rhs_pat(i);
  }
  /* Call the DEPLOYED bundle kernel via the generated header prototype. */
  tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(
      lhs, rhs, &acc, scale, &out, n);
  float expected = scalar_oracle(lhs, rhs, acc0, scale, n);
  float abs_err = fabsf(out - expected);
  int ok = abs_err <= 1e-05f * (1.0f + fabsf(expected));
  printf("n=%zu acc0=%d scale=%g bundle=%.6f scalar=%.6f abs_err=%.3e %s\n",
         n, acc0, scale, out, expected, abs_err, ok ? "OK" : "MISMATCH");
  free(lhs);
  free(rhs);
  return ok;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 256, 257, 1024, 4096, 16384, 65536};
  const float scales[] = {-0.125f, 0.375f};
  const int32_t accs[] = {0, 7, -19};
  int all_ok = 1;
  for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    for (size_t s = 0; s < sizeof(scales) / sizeof(scales[0]); ++s)
      for (size_t a = 0; a < sizeof(accs) / sizeof(accs[0]); ++a)
        all_ok &= run_case(counts[c], accs[a], scales[s]);
  printf("%s\n", all_ok ? "DEPLOYED_BUNDLE_ALL_PASS" : "DEPLOYED_BUNDLE_FAIL");
  return all_ok ? 0 : 1;
}
