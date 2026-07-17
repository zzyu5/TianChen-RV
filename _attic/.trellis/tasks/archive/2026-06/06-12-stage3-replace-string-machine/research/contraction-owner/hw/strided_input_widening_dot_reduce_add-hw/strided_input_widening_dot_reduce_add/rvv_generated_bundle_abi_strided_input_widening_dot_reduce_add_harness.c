#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, size_t lhs_stride, size_t rhs_stride,
                    int pattern) {
  /* expected: (int32_t)(acc[0] + sum_i((int32_t)lhs[i * lhs_stride] * (int32_t)rhs[i * rhs_stride])) */
  size_t lhs_alloc = n * lhs_stride + 8;
  size_t rhs_alloc = n * rhs_stride + 8;
  size_t out_alloc = n + 5;
  if (out_alloc == 5 && n == 0)
    out_alloc = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * lhs_alloc);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * rhs_alloc);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  if (!lhs || !rhs || !acc || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < lhs_alloc; ++index)
    lhs[index] = (int16_t)((pattern == 0)
                               ? (((index % 2) == 0) ? 17 : -19)
                               : (((index % 3) == 0) ? -31 : 37));
  for (size_t index = 0; index < rhs_alloc; ++index)
    rhs[index] = (int16_t)((pattern == 0)
                               ? (((index % 3) == 0) ? -23 : 29)
                               : (((index % 4) == 0) ? 41 : -43));
  for (size_t index = 0; index < out_alloc; ++index) {
    acc[index] = (int32_t)((int32_t)31 + pattern * 7);
    out[index] = (int32_t)0x5a5a5a5a;
  }
  for (size_t index = 0; index < n; ++index) {
    lhs[index * lhs_stride] =
        (int16_t)((pattern == 0)
                      ? (((index % 4) < 2) ? 200 : -210)
                      : (((index % 6) < 3) ? -220 : 230));
    rhs[index * rhs_stride] =
        (int16_t)((pattern == 0)
                      ? (((index % 5) == 0) ? -190 : 180)
                      : (((index % 5) < 2) ? 170 : -185));
  }

  tcrv_emitc_explicit_strided_dot_kernel_rvv_explicit_strided_input_dot(lhs, rhs, acc, out, n, lhs_stride, rhs_stride);

  int32_t expected = acc[0];
  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t widening_products = 0;
  size_t lhs_skipped_nonzero = 0;
  size_t rhs_skipped_nonzero = 0;
  for (size_t index = 0; index < n; ++index) {
    size_t lhs_index = index * lhs_stride;
    size_t rhs_index = index * rhs_stride;
    int32_t product = (int32_t)lhs[lhs_index] * (int32_t)rhs[rhs_index];
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (product > 32767 || product < -32768)
      ++widening_products;
    expected = (int32_t)(expected + product);
  }
  for (size_t index = 0; index < n * lhs_stride; ++index)
    if ((index % lhs_stride) != 0 && lhs[index] != 0)
      ++lhs_skipped_nonzero;
  for (size_t index = 0; index < n * rhs_stride; ++index)
    if ((index % rhs_stride) != 0 && rhs[index] != 0)
      ++rhs_skipped_nonzero;

  if (out[0] != expected) {
    fprintf(stderr,
            "strided_input_widening_dot_reduce_add scalar mismatch n=%zu got=%d expected=%d seed=%d lhs_stride=%zu rhs_stride=%zu pattern=%d positive_products=%zu negative_products=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu\n",
            n, out[0], expected, acc[0], lhs_stride, rhs_stride,
            pattern, positive_products, negative_products,
            lhs_skipped_nonzero, rhs_skipped_nonzero);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }

  for (size_t index = 1; index < out_alloc; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "strided_input_widening_dot_reduce_add touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\n",
              n, index, out[index], (int32_t)0x5a5a5a5a);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }
  }

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                widening_products == 0 || lhs_skipped_nonzero == 0 ||
                rhs_skipped_nonzero == 0 || acc[0] == 0)) {
    fprintf(stderr,
            "strided_input_widening_dot_reduce_add coverage missing n=%zu lhs_stride=%zu rhs_stride=%zu pattern=%d positive_products=%zu negative_products=%zu widening_products=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu seed=%d\n",
            n, lhs_stride, rhs_stride, pattern, positive_products,
            negative_products, widening_products, lhs_skipped_nonzero,
            rhs_skipped_nonzero, acc[0]);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("strided_input_widening_dot_reduce_add case n=%zu ok strided_signed_horizontal_dot seed_added widening_products=%zu source_strides=%zu,%zu data_pattern=%d skipped_source_elements_ignored scalar_output tail_preserved\n",
         n, widening_products, lhs_stride, rhs_stride, pattern);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const struct {
    size_t lhs_stride;
    size_t rhs_stride;
    int pattern;
  } cases[] = {
      {2, 3, 0},
      {3, 2, 1},
  };
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t case_count = sizeof(cases) / sizeof(cases[0]);
  for (size_t index = 0; index < count_count; ++index) {
    for (size_t case_index = 0; case_index < case_count; ++case_index) {
      int status = run_case(counts[index], cases[case_index].lhs_stride,
                            cases[case_index].rhs_stride,
                            cases[case_index].pattern);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_strided_input_widening_dot_reduce_add_ok counts=0,1,16,17,257 stride_pairs=2:3,3:2 data_patterns=2\n");
  printf("PASS op=strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 data_patterns=2\n");
  return 0;
}
