#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int32_t init_cmp_lhs(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 4) == 0 || (index % 4) == 3) ? (int32_t)(10 + (int32_t)index) : (int32_t)(100 + (int32_t)index));
  return (int32_t)(((index % 5) < 2)
                       ? (int32_t)(5 + (int32_t)index)
                       : (int32_t)(90 + (int32_t)index));
}

static int32_t init_cmp_rhs(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 4) == 0 || (index % 4) == 3) ? (int32_t)(50 + (int32_t)index) : (int32_t)(20 + (int32_t)index));
  return (int32_t)(((index % 5) < 2)
                       ? (int32_t)(60 + (int32_t)index)
                       : (int32_t)(30 + (int32_t)index));
}

static int32_t init_lhs(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 2) == 0) ? ((int32_t)(index % 7) + 2) : -((int32_t)(index % 7) + 2));
  return (int32_t)(((index % 3) == 0)
                       ? -((int32_t)(index % 13) + 4)
                       : ((int32_t)(index % 13) + 5));
}

static int32_t init_rhs(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 3) == 0) ? -((int32_t)(index % 5) + 3) : ((int32_t)(index % 5) + 3));
  return (int32_t)(((index % 4) < 2)
                       ? ((int32_t)(index % 17) + 6)
                       : -((int32_t)(index % 17) + 7));
}

static int32_t init_acc(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 2) == 0) ? (23 - (int32_t)(index % 11)) : -(23 - (int32_t)(index % 11)));
  return (int32_t)(((index % 2) == 0)
                       ? (31 - (int32_t)(index % 17))
                       : -(29 + (int32_t)(index % 17)));
}

static int run_case(size_t n, int pattern) {
  /* expected: (cmp_lhs[index] < cmp_rhs[index] ? (int32_t)(acc[index] + (int32_t)(lhs[index] * rhs[index])) : acc[index]) */
  size_t alloc_n = n + 8;
  if (alloc_n == 8 && n == 0)
    alloc_n = 9;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
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
    cmp_lhs[index] = init_cmp_lhs(index, pattern);
    cmp_rhs[index] = init_cmp_rhs(index, pattern);
    lhs[index] = init_lhs(index, pattern);
    rhs[index] = init_rhs(index, pattern);
    acc[index] = init_acc(index, pattern);
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_computed_masked_macc_add_kernel_explicit_selected_body_rvv_computed_masked_macc_add(cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_acc_preserved = 0;
  size_t add_only_distinguishing = 0;
  size_t mul_only_distinguishing = 0;
  size_t signed_product_lanes = 0;
  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t positive_accumulators = 0;
  size_t negative_accumulators = 0;
  for (size_t index = 0; index < n; ++index) {
    int predicate = cmp_lhs[index] < cmp_rhs[index];
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = (cmp_lhs[index] < cmp_rhs[index] ? (int32_t)(acc[index] + (int32_t)(lhs[index] * rhs[index])) : acc[index]);
    int32_t add_only = (int32_t)(acc[index] + lhs[index] + rhs[index]);
    int32_t mul_only = product;
    if (predicate)
      ++active_lanes;
    else
      ++inactive_lanes;
    if (!predicate && out[index] == acc[index])
      ++inactive_acc_preserved;
    if (predicate && expected != add_only)
      ++add_only_distinguishing;
    if (predicate && expected != mul_only)
      ++mul_only_distinguishing;
    if (predicate && product != 0 && acc[index] != 0)
      ++signed_product_lanes;
    if (predicate && product > 0)
      ++positive_products;
    if (predicate && product < 0)
      ++negative_products;
    if (acc[index] > 0)
      ++positive_accumulators;
    if (acc[index] < 0)
      ++negative_accumulators;
    if (out[index] != expected) {
      fprintf(stderr,
              "computed_masked_macc_add mismatch n=%zu pattern=%d index=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d lhs=%d rhs=%d acc=%d predicate=%d product=%d\n",
              n, pattern, index, out[index], expected, cmp_lhs[index], cmp_rhs[index],
              lhs[index], rhs[index], acc[index], predicate, product);
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 12;
    }
  }

  for (size_t index = n; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "computed_masked_macc_add touched tail sentinel n=%zu pattern=%d raw_index=%zu got=%d sentinel=%d\n",
              n, pattern, index, out[index], (int32_t)0x5a5a5a5a);
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (cmp_lhs[index] != init_cmp_lhs(index, pattern) ||
        cmp_rhs[index] != init_cmp_rhs(index, pattern) ||
        lhs[index] != init_lhs(index, pattern) ||
        rhs[index] != init_rhs(index, pattern) ||
        acc[index] != init_acc(index, pattern)) {
      fprintf(stderr,
              "computed_masked_macc_add source buffer mutated n=%zu pattern=%d raw_index=%zu\n",
              n, pattern, index);
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 14;
    }
  }

  if (n > 3 && (active_lanes == 0 || inactive_lanes == 0 ||
                inactive_acc_preserved == 0 ||
                add_only_distinguishing == 0 ||
                mul_only_distinguishing == 0 ||
                signed_product_lanes == 0 ||
                positive_products == 0 || negative_products == 0 ||
                positive_accumulators == 0 || negative_accumulators == 0)) {
    fprintf(stderr,
            "computed_masked_macc_add coverage missing n=%zu pattern=%d active=%zu inactive=%zu inactive_acc_preserved=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu signed_product_lanes=%zu positive_products=%zu negative_products=%zu positive_accumulators=%zu negative_accumulators=%zu\n",
            n, pattern, active_lanes, inactive_lanes, inactive_acc_preserved,
            add_only_distinguishing, mul_only_distinguishing,
            signed_product_lanes, positive_products, negative_products,
            positive_accumulators, negative_accumulators);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 15;
  }

  free(cmp_lhs);
  free(cmp_rhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("computed_masked_macc_add case n=%zu pattern=%d ok computed_mask macc active_lanes=%zu inactive_lanes=%zu inactive_acc_preserved=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu signed_products=%zu source_preserved tail_preserved\n",
         n, pattern, active_lanes, inactive_lanes, inactive_acc_preserved,
         add_only_distinguishing, mul_only_distinguishing,
         signed_product_lanes);
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
  printf("tcrv_rvv_generated_bundle_abi_computed_masked_macc_add_ok counts=0,1,16,17,257 patterns=0,1\n");
  printf("PASS op=computed_masked_macc_add counts=0,1,16,17,257 patterns=0,1\n");
  return 0;
}
