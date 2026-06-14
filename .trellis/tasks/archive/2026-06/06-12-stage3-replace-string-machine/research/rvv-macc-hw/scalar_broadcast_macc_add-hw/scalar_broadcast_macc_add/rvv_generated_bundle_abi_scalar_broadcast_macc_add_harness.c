#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int32_t rhs_scalar) {
  /* expected: (int32_t)(acc[index] + (int32_t)(lhs[index] * rhs_scalar)) */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !acc || !out) {
    fprintf(stderr, "allocation failed for n=%zu rhs_scalar=%d\n", n, rhs_scalar);
    free(lhs);
    free(acc);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    lhs[index] = (int32_t)(((index % 2) == 0) ? ((int32_t)(index % 7) + 1) : -((int32_t)(index % 7) + 1));
    acc[index] = (int32_t)(((index % 2) == 0) ? (17 - (int32_t)(index % 9)) : -(17 - (int32_t)(index % 9)));
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_scalar_broadcast_macc_add_kernel_explicit_selected_body_rvv_scalar_broadcast_macc_add(lhs, rhs_scalar, acc, out, n);

  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t positive_lhs = 0;
  size_t negative_lhs = 0;
  size_t positive_accumulators = 0;
  size_t negative_accumulators = 0;
  size_t nonzero_accumulators = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs[index] * rhs_scalar;
    int32_t expected = (int32_t)(acc[index] + (int32_t)(lhs[index] * rhs_scalar));
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (lhs[index] > 0)
      ++positive_lhs;
    if (lhs[index] < 0)
      ++negative_lhs;
    if (acc[index] > 0)
      ++positive_accumulators;
    if (acc[index] < 0)
      ++negative_accumulators;
    if (acc[index] != 0)
      ++nonzero_accumulators;
    if (out[index] != expected) {
      fprintf(stderr,
              "scalar_broadcast_macc_add mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs_scalar=%d acc=%d product=%d\n",
              n, index, out[index], expected, lhs[index], rhs_scalar, acc[index], product);
      free(lhs);
      free(acc);
      free(out);
      return 12;
    }
  }

  for (size_t index = n; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "scalar_broadcast_macc_add touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar=%d\n",
              n, index, out[index], (int32_t)0x5a5a5a5a, rhs_scalar);
      free(lhs);
      free(acc);
      free(out);
      return 13;
    }
  }

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                positive_lhs == 0 || negative_lhs == 0 ||
                positive_accumulators == 0 || negative_accumulators == 0 ||
                nonzero_accumulators == 0)) {
    fprintf(stderr,
            "scalar_broadcast_macc_add coverage missing n=%zu rhs_scalar=%d positive_products=%zu negative_products=%zu positive_lhs=%zu negative_lhs=%zu positive_accumulators=%zu negative_accumulators=%zu nonzero_accumulators=%zu\n",
            n, rhs_scalar, positive_products, negative_products, positive_lhs,
            negative_lhs, positive_accumulators, negative_accumulators,
            nonzero_accumulators);
    free(lhs);
    free(acc);
    free(out);
    return 14;
  }

  free(lhs);
  free(acc);
  free(out);
  printf("scalar_broadcast_macc_add case n=%zu ok rhs_scalar=%d scalar_broadcast_macc explicit_accumulator signed_products tail_preserved\n",
         n, rhs_scalar);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const int32_t rhs_scalar_values[] = {(int32_t)-37, (int32_t)91};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {
    for (size_t index = 0; index < count_count; ++index) {
      int status = run_case(counts[index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_scalar_broadcast_macc_add_ok counts=0,1,16,17,257 rhs_scalars=-37,91\n");
  printf("PASS op=scalar_broadcast_macc_add counts=0,1,16,17,257 rhs_scalars=-37,91\n");
  return 0;
}
