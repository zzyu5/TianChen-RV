#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n) {
  /* expected: (int32_t)(acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? (int32_t)dot_lhs[i] * (int32_t)dot_rhs[i] : 0)) */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!cmp_lhs || !cmp_rhs || !lhs || !rhs || !acc || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    cmp_lhs[index] = (int32_t)(((index % 4) == 0 || (index % 4) == 3) ? (int32_t)(10 + (int32_t)index) : (int32_t)(100 + (int32_t)index));
    cmp_rhs[index] = (int32_t)(((index % 4) == 0 || (index % 4) == 3) ? (int32_t)(50 + (int32_t)index) : (int32_t)(20 + (int32_t)index));
    lhs[index] = (int16_t)(((index % 4) < 2)
                               ? -((int)(index % 53) + 2)
                               : ((int)(index % 53) + 5));
    rhs[index] = (int16_t)(((index % 5) == 0)
                               ? -((int)(index % 37) + 3)
                               : ((int)(index % 37) + 7));
    acc[index] = (int32_t)29;
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_masked_wdot_kernel_rvv_explicit_masked_wdot(cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n);

  int32_t expected = acc[0];
  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t active_positive_products = 0;
  size_t active_negative_products = 0;
  size_t inactive_nonzero_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int active = cmp_lhs[index] < cmp_rhs[index];
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    if (active) {
      ++active_lanes;
      if (product > 0)
        ++active_positive_products;
      if (product < 0)
        ++active_negative_products;
      expected = (int32_t)(expected + product);
    } else {
      ++inactive_lanes;
      if (product != 0)
        ++inactive_nonzero_products;
    }
  }

  if (out[0] != expected) {
    fprintf(stderr,
            "computed_masked_widening_dot_reduce_add scalar mismatch n=%zu got=%d expected=%d seed=%d active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu\n",
            n, out[0], expected, acc[0], active_lanes, inactive_lanes,
            active_positive_products, active_negative_products,
            inactive_nonzero_products);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }

  for (size_t index = 1; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "computed_masked_widening_dot_reduce_add touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\n",
              n, index, out[index], (int32_t)0x5a5a5a5a);
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }
  }

  if (n > 4 && (active_lanes == 0 || inactive_lanes == 0 ||
                active_positive_products == 0 ||
                active_negative_products == 0 ||
                inactive_nonzero_products == 0 || acc[0] == 0)) {
    fprintf(stderr,
            "computed_masked_widening_dot_reduce_add coverage missing n=%zu active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu seed=%d\n",
            n, active_lanes, inactive_lanes, active_positive_products,
            active_negative_products, inactive_nonzero_products, acc[0]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }

  free(cmp_lhs);
  free(cmp_rhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("computed_masked_widening_dot_reduce_add case n=%zu ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved\n", n);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  for (size_t index = 0; index < count_count; ++index) {
    int status = run_case(counts[index]);
    if (status != 0)
      return status;
  }
  printf("tcrv_rvv_generated_bundle_abi_computed_masked_widening_dot_reduce_add_ok counts=0,1,16,17,257\n");
  printf("PASS op=computed_masked_widening_dot_reduce_add counts=0,1,16,17,257\n");
  return 0;
}
