#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int pattern) {
  /* expected: (int32_t)(acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i])) */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int8_t *lhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int8_t *lhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
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
      lhs[index] = (int8_t)(((index % 4) < 2) ? -((int)(index % 47) + 12) : ((int)(index % 43) + 14));
      rhs[index] = (int8_t)(((index % 5) == 0) ? -((int)(index % 31) + 15) : ((int)(index % 29) + 17));
      acc[index] = (int32_t)19;
    } else {
      lhs[index] = (int8_t)(((index % 3) == 0)
                               ? -((int)(index % 23) + 5)
                               : ((int)(index % 19) + 7));
      rhs[index] = (int8_t)(((index % 4) < 2)
                               ? ((int)(index % 17) + 9)
                               : -((int)(index % 13) + 11));
      acc[index] = (int32_t)(((index % 2) == 0)
                                ? -53 - (int32_t)(index * 5)
                                : 71 + (int32_t)(index * 7));
    }
    lhs_before[index] = lhs[index];
    rhs_before[index] = rhs[index];
    acc_before[index] = acc[index];
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_product_reduce_kernel_explicit_selected_body_rvv_product_reduce(lhs, rhs, acc, out, n);

  int32_t expected = acc[0];
  int32_t add_only_expected = acc[0];
  int32_t no_seed_expected = 0;
  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t i16_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    add_only_expected = (int32_t)(add_only_expected + (int32_t)lhs[index] +
                                  (int32_t)rhs[index]);
    no_seed_expected = (int32_t)(no_seed_expected + product);
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (product > 127 || product < -128)
      ++i16_products;
    expected = (int32_t)(expected + product);
  }

  if (out[0] != expected) {
    fprintf(stderr,
            "widening_product_reduce_add scalar mismatch n=%zu pattern=%d got=%d expected=%d seed=%d positive_products=%zu negative_products=%zu i16_products=%zu\n",
            n, pattern, out[0], expected, acc[0], positive_products,
            negative_products, i16_products);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    free(lhs_before);
    free(rhs_before);
    free(acc_before);
    return 12;
  }

  for (size_t index = 1; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "widening_product_reduce_add touched non-scalar/tail sentinel n=%zu pattern=%d raw_index=%zu got=%d sentinel=%d\n",
              n, pattern, index, out[index], (int32_t)0x5a5a5a5a);
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

  size_t add_only_distinguishing = expected != add_only_expected;
  size_t mul_only_distinguishing = expected != no_seed_expected;
  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                i16_products == 0 || acc[0] == 0 ||
                add_only_distinguishing == 0 ||
                mul_only_distinguishing == 0)) {
    fprintf(stderr,
            "widening_product_reduce_add coverage missing n=%zu pattern=%d positive_products=%zu negative_products=%zu i16_products=%zu seed=%d add_only_distinguishing=%zu mul_only_distinguishing=%zu\n",
            n, pattern, positive_products, negative_products, i16_products,
            acc[0], add_only_distinguishing, mul_only_distinguishing);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    free(lhs_before);
    free(rhs_before);
    free(acc_before);
    return 14;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index] ||
        acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_add source buffer mutated n=%zu pattern=%d index=%zu lhs=%d expected_lhs=%d rhs=%d expected_rhs=%d acc=%d expected_acc=%d\n",
              n, pattern, index, lhs[index], lhs_before[index], rhs[index],
              rhs_before[index], acc[index], acc_before[index]);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      free(lhs_before);
      free(rhs_before);
      free(acc_before);
      return 15;
    }
  }

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  free(lhs_before);
  free(rhs_before);
  free(acc_before);
  printf("widening_product_reduce_add case n=%zu pattern=%d ok signed_i8_products i16_product_intermediate seed_added scalar_output tail_preserved i16_products=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu\n",
         n, pattern, i16_products, add_only_distinguishing,
         mul_only_distinguishing);
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
  printf("tcrv_rvv_generated_bundle_abi_widening_product_reduce_add_ok counts=0,1,16,17,257 patterns=0,1\n");
  printf("PASS op=widening_product_reduce_add counts=0,1,16,17,257 patterns=0,1\n");
  return 0;
}
