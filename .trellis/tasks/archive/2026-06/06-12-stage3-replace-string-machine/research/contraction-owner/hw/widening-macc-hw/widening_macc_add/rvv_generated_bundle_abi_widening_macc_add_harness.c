#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int pattern) {
  /* expected: (int32_t)(acc[index] + (int32_t)lhs[index] * (int32_t)rhs[index]) */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int16_t *lhs_before = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs_before = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc_before = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !acc || !out || !lhs_before || !rhs_before || !acc_before) {
    fprintf(stderr, "allocation failed for n=%zu pattern=%d\n", n, pattern);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    free(lhs_before);
    free(rhs_before);
    free(acc_before);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = (int16_t)(((index % 4) < 2) ? -((int)(index % 257) + 260) : ((int)(index % 251) + 280));
      rhs[index] = (int16_t)(((index % 5) == 0) ? -((int)(index % 191) + 170) : ((int)(index % 181) + 190));
      acc[index] = (int32_t)(300 + (int32_t)(index * 11));
    } else {
      lhs[index] = (int16_t)(((index % 3) == 0)
                                ? ((int)(index % 157) + 240)
                                : -((int)(index % 149) + 230));
      rhs[index] = (int16_t)(((index % 4) < 2)
                                ? -((int)(index % 181) + 180)
                                : ((int)(index % 167) + 210));
      acc[index] = (int32_t)(((index % 2) == 0)
                                ? -700 - (int32_t)(index * 13)
                                : 900 + (int32_t)(index * 17));
    }
    lhs_before[index] = lhs[index];
    rhs_before[index] = rhs[index];
    acc_before[index] = acc[index];
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_widening_macc_add_kernel_explicit_selected_body_rvv_widening_macc_add(lhs, rhs, acc, out, n);

  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t widening_products = 0;
  size_t nonzero_accumulators = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = (int32_t)(acc[index] + (int32_t)lhs[index] * (int32_t)rhs[index]);
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (product > INT16_MAX || product < INT16_MIN)
      ++widening_products;
    if (acc[index] != 0)
      ++nonzero_accumulators;
    if (out[index] != expected) {
      fprintf(stderr,
              "widening_macc_add mismatch n=%zu pattern=%d index=%zu got=%d expected=%d lhs=%d rhs=%d acc=%d product=%d\n",
              n, pattern, index, out[index], expected, lhs[index], rhs[index], acc[index], product);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      free(lhs_before);
      free(rhs_before);
      free(acc_before);
      return 12;
    }
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] ||
        rhs[index] != rhs_before[index] ||
        acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_macc_add source or accumulator buffer mutated n=%zu pattern=%d index=%zu lhs=%d before_lhs=%d rhs=%d before_rhs=%d acc=%d before_acc=%d\n",
              n, pattern, index, lhs[index], lhs_before[index], rhs[index], rhs_before[index], acc[index], acc_before[index]);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      free(lhs_before);
      free(rhs_before);
      free(acc_before);
      return 13;
    }
  }

  for (size_t index = n; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "widening_macc_add touched tail sentinel n=%zu pattern=%d raw_index=%zu got=%d sentinel=%d\n",
              n, pattern, index, out[index], (int32_t)0x5a5a5a5a);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      free(lhs_before);
      free(rhs_before);
      free(acc_before);
      return 14;
    }
  }

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                widening_products == 0 ||
                nonzero_accumulators == 0)) {
    fprintf(stderr,
            "widening_macc_add coverage missing n=%zu pattern=%d positive_products=%zu negative_products=%zu widening_products=%zu nonzero_accumulators=%zu\n",
            n, pattern, positive_products, negative_products, widening_products, nonzero_accumulators);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    free(lhs_before);
    free(rhs_before);
    free(acc_before);
    return 15;
  }

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  free(lhs_before);
  free(rhs_before);
  free(acc_before);
  printf("widening_macc_add case n=%zu pattern=%d ok signed_widening_products accumulation source_preserved accumulator_preserved tail_preserved positive_products=%zu negative_products=%zu widening_products=%zu\n",
         n, pattern, positive_products, negative_products, widening_products);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  for (size_t index = 0; index < count_count; ++index) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      int status = run_case(counts[index], pattern);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_widening_macc_add_ok counts=0,1,16,17,257 patterns=0,1\n");
  printf("PASS op=widening_macc_add counts=0,1,16,17,257 patterns=0,1\n");
  return 0;
}
