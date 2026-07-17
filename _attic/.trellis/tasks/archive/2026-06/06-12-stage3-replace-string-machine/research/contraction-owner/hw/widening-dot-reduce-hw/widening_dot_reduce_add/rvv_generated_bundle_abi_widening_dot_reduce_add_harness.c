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
      acc[index] = (int32_t)17;
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

  tcrv_emitc_explicit_selected_body_widening_dot_reduce_add_kernel_explicit_selected_body_rvv_widening_dot_reduce_add(lhs, rhs, acc, out, n);

  int32_t expected = acc[0];
  int32_t add_only_expected = acc[0];
  int32_t no_seed_expected = 0;
  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t widening_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    add_only_expected = (int32_t)(add_only_expected + (int32_t)lhs[index] +
                                  (int32_t)rhs[index]);
    no_seed_expected = (int32_t)(no_seed_expected + product);
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (product > 32767 || product < -32768)
      ++widening_products;
    expected = (int32_t)(expected + product);
  }

  if (out[0] != expected) {
    fprintf(stderr,
            "widening_dot_reduce_add scalar mismatch n=%zu got=%d expected=%d seed=%d positive_products=%zu negative_products=%zu\n",
            n, out[0], expected, acc[0], positive_products, negative_products);
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
              "widening_dot_reduce_add touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\n",
              n, index, out[index], (int32_t)0x5a5a5a5a);
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

  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] ||
        rhs[index] != rhs_before[index] ||
        acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_dot_reduce_add source or accumulator buffer mutated n=%zu index=%zu lhs=%d before_lhs=%d rhs=%d before_rhs=%d acc=%d before_acc=%d\n",
              n, index, lhs[index], lhs_before[index], rhs[index],
              rhs_before[index], acc[index], acc_before[index]);
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

  size_t add_only_distinguishing = expected != add_only_expected;
  size_t mul_only_distinguishing = expected != no_seed_expected;
  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                widening_products == 0 || acc[0] == 0 ||
                add_only_distinguishing == 0 ||
                mul_only_distinguishing == 0)) {
    fprintf(stderr,
            "widening_dot_reduce_add coverage missing n=%zu positive_products=%zu negative_products=%zu widening_products=%zu seed=%d add_only_distinguishing=%zu mul_only_distinguishing=%zu\n",
            n, positive_products, negative_products, widening_products, acc[0],
            add_only_distinguishing, mul_only_distinguishing);
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
  printf("widening_dot_reduce_add case n=%zu ok signed_horizontal_dot seed_added scalar_output source_preserved accumulator_preserved tail_preserved widening_products=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu\n",
         n, widening_products, add_only_distinguishing, mul_only_distinguishing);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const int patterns[] = {0, 1};
  const size_t pattern_count = sizeof(patterns) / sizeof(patterns[0]);
  for (size_t index = 0; index < count_count; ++index) {
    for (size_t pattern_index = 0; pattern_index < pattern_count;
         ++pattern_index) {
      int pattern = patterns[pattern_index];
      int status = run_case(counts[index], pattern);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_widening_dot_reduce_add_ok counts=0,1,16,17,257 patterns=0,1\n");
  printf("PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1\n");
  return 0;
}
