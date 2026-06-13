#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

typedef struct {
  size_t mask_a_true;
  size_t mask_a_false;
  size_t mask_b_true;
  size_t mask_b_false;
  size_t composed_true;
  size_t composed_false;
  size_t single_mask_only;
  size_t true_payload_lanes;
  size_t false_payload_lanes;
} dual_compare_select_coverage_t;

static int run_case(size_t n, int32_t rhs_scalar_a,
                    int32_t rhs_scalar_b,
                    dual_compare_select_coverage_t *coverage) {
  /* expected: ((cmp_lhs_a[index] <= rhs_scalar_a && cmp_lhs_b[index] <= rhs_scalar_b) ? true_value[index] : false_value[index]) */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs_a = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_lhs_b = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *true_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *false_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!cmp_lhs_a || !cmp_lhs_b || !true_value || !false_value || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(cmp_lhs_a);
    free(cmp_lhs_b);
    free(true_value);
    free(false_value);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    cmp_lhs_a[index] = (int32_t)(((index % 6) == 0) ? -100 : ((index % 6) == 1) ? -37 : ((index % 6) == 2) ? -12 : ((index % 6) == 3) ? 0 : ((index % 6) == 4) ? 91 : 130);
    cmp_lhs_b[index] = (int32_t)(((index % 6) == 0) ? -50 :
                         ((index % 6) == 1) ? -10 :
                         ((index % 6) == 2) ? -60 :
                         ((index % 6) == 3) ? 100 :
                         ((index % 6) == 4) ? 50 : 120);
    true_value[index] = (int32_t)(5100 + (int32_t)(index * 31));
    false_value[index] = (int32_t)(-6100 - (int32_t)(index * 37));
    out[index] = (int32_t)0x5a5a5a5a;
  }
  for (size_t index = alloc_n; index < out_alloc_n; ++index)
    out[index] = (int32_t)0x5a5a5a5a;

  tcrv_emitc_explicit_dual_cmp_mask_select_kernel_explicit_rvv_dual_cmp_mask_select(cmp_lhs_a, rhs_scalar_a, cmp_lhs_b, rhs_scalar_b,
                              true_value, false_value, out, n);

  size_t mask_a_true = 0;
  size_t mask_a_false = 0;
  size_t mask_b_true = 0;
  size_t mask_b_false = 0;
  size_t composed_true = 0;
  size_t composed_false = 0;
  size_t single_mask_only = 0;
  size_t true_payload_lanes = 0;
  size_t false_payload_lanes = 0;
  for (size_t index = 0; index < n; ++index) {
    int predicate_a = cmp_lhs_a[index] <= rhs_scalar_a;
    int predicate_b = cmp_lhs_b[index] <= rhs_scalar_b;
    int composed = predicate_a && predicate_b;
    if (predicate_a)
      ++mask_a_true;
    else
      ++mask_a_false;
    if (predicate_b)
      ++mask_b_true;
    else
      ++mask_b_false;
    if (composed)
      ++composed_true;
    else
      ++composed_false;
    if (predicate_a != predicate_b)
      ++single_mask_only;

    int32_t expected = ((cmp_lhs_a[index] <= rhs_scalar_a && cmp_lhs_b[index] <= rhs_scalar_b) ? true_value[index] : false_value[index]);
    if (expected == true_value[index])
      ++true_payload_lanes;
    if (expected == false_value[index])
      ++false_payload_lanes;
    if (out[index] != expected) {
      fprintf(stderr,
              "runtime_scalar_dual_cmp_mask_and_select mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs_a=%d rhs_scalar_a=%d cmp_lhs_b=%d rhs_scalar_b=%d true=%d false=%d mask_a=%d mask_b=%d composed=%d\n",
              n, index, out[index],
              expected, cmp_lhs_a[index],
              rhs_scalar_a,
              cmp_lhs_b[index],
              rhs_scalar_b,
              true_value[index],
              false_value[index], predicate_a, predicate_b,
              composed);
      free(cmp_lhs_a);
      free(cmp_lhs_b);
      free(true_value);
      free(false_value);
      free(out);
      return 12;
    }
  }

  for (size_t index = n; index < out_alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "runtime_scalar_dual_cmp_mask_and_select touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar_a=%d rhs_scalar_b=%d\n",
              n, index, out[index],
              (int32_t)0x5a5a5a5a,
              rhs_scalar_a,
              rhs_scalar_b);
      free(cmp_lhs_a);
      free(cmp_lhs_b);
      free(true_value);
      free(false_value);
      free(out);
      return 13;
    }
  }

  if (coverage) {
    coverage->mask_a_true += mask_a_true;
    coverage->mask_a_false += mask_a_false;
    coverage->mask_b_true += mask_b_true;
    coverage->mask_b_false += mask_b_false;
    coverage->composed_true += composed_true;
    coverage->composed_false += composed_false;
    coverage->single_mask_only += single_mask_only;
    coverage->true_payload_lanes += true_payload_lanes;
    coverage->false_payload_lanes += false_payload_lanes;
  }

  free(cmp_lhs_a);
  free(cmp_lhs_b);
  free(true_value);
  free(false_value);
  free(out);
  printf("runtime_scalar_dual_cmp_mask_and_select case n=%zu ok rhs_scalar_a=%d rhs_scalar_b=%d mask_a_true=%zu mask_b_true=%zu composed_true=%zu single_mask_only=%zu tail_preserved\n",
         n, rhs_scalar_a,
         rhs_scalar_b, mask_a_true, mask_b_true,
         composed_true, single_mask_only);
  return 0;
}

int main(void) {
  const size_t counts[] = {1, 7, 16, 17, 257};
  const int32_t rhs_scalar_a_values[] = {(int32_t)-37, (int32_t)91};
  const int32_t rhs_scalar_b_values[] = {(int32_t)-37, (int32_t)91};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_a_count = sizeof(rhs_scalar_a_values) / sizeof(rhs_scalar_a_values[0]);
  const size_t scalar_b_count = sizeof(rhs_scalar_b_values) / sizeof(rhs_scalar_b_values[0]);
  dual_compare_select_coverage_t coverage = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (size_t scalar_a_index = 0; scalar_a_index < scalar_a_count; ++scalar_a_index) {
    for (size_t scalar_b_index = 0; scalar_b_index < scalar_b_count; ++scalar_b_index) {
      for (size_t index = 0; index < count_count; ++index) {
        int status = run_case(counts[index], rhs_scalar_a_values[scalar_a_index],
                              rhs_scalar_b_values[scalar_b_index], &coverage);
        if (status != 0)
          return status;
      }
    }
  }
  if (coverage.mask_a_true == 0 || coverage.mask_a_false == 0 ||
      coverage.mask_b_true == 0 || coverage.mask_b_false == 0 ||
      coverage.composed_true == 0 || coverage.composed_false == 0) {
    fprintf(stderr,
            "runtime_scalar_dual_cmp_mask_and_select aggregate mask coverage missing mask_a_true=%zu mask_a_false=%zu mask_b_true=%zu mask_b_false=%zu composed_true=%zu composed_false=%zu\n",
            coverage.mask_a_true, coverage.mask_a_false, coverage.mask_b_true,
            coverage.mask_b_false, coverage.composed_true,
            coverage.composed_false);
    return 14;
  }
  if (coverage.single_mask_only == 0) {
    fprintf(stderr,
            "runtime_scalar_dual_cmp_mask_and_select aggregate mask-and distinction missing\n");
    return 15;
  }
  if (coverage.true_payload_lanes == 0 || coverage.false_payload_lanes == 0) {
    fprintf(stderr,
            "runtime_scalar_dual_cmp_mask_and_select aggregate select payload coverage missing true_lanes=%zu false_lanes=%zu\n",
            coverage.true_payload_lanes, coverage.false_payload_lanes);
    return 16;
  }
  printf("tcrv_rvv_generated_bundle_abi_runtime_scalar_dual_cmp_mask_and_select_ok counts=1,7,16,17,257 rhs_scalar_a_values=-37,91 rhs_scalar_b_values=-37,91\n");
  printf("PASS op=runtime_scalar_dual_cmp_mask_and_select counts=1,7,16,17,257 rhs_scalar_a_values=-37,91 rhs_scalar_b_values=-37,91\n");
  return 0;
}
