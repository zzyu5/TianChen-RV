#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int pattern, float scale) {
  /* expected: ((float)(acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i]))) * scale */
  const float tolerance = 1e-05f;
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = 16;
  size_t acc_alloc_n = 4;
  int8_t *lhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *lhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * acc_alloc_n);
  int32_t *acc_before = (int32_t *)malloc(sizeof(int32_t) * acc_alloc_n);
  float *out = (float *)malloc(sizeof(float) * out_alloc_n);
  float *old_out = (float *)malloc(sizeof(float) * out_alloc_n);
  int status = 0;
  if (!lhs || !rhs || !lhs_before || !rhs_before || !acc || !acc_before ||
      !out || !old_out) {
    fprintf(stderr,
            "allocation failed for n=%zu pattern=%d scale=%.9g\n",
            n, pattern, scale);
    status = 11;
    goto cleanup;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = (int8_t)(((index % 4) < 2) ? -((int)(index % 47) + 12) : ((int)(index % 43) + 14));
      rhs[index] = (int8_t)(((index % 5) == 0) ? -((int)(index % 31) + 15) : ((int)(index % 29) + 17));
    } else {
      lhs[index] = (int8_t)(((index % 6) < 3)
                                ? -((int)(index % 53) + 21)
                                : ((int)(index % 47) + 18));
      rhs[index] = (int8_t)(((index % 5) == 1 || (index % 5) == 4)
                                ? -((int)(index % 43) + 17)
                                : ((int)(index % 41) + 23));
    }
    lhs_before[index] = lhs[index];
    rhs_before[index] = rhs[index];
  }
  for (size_t index = 0; index < acc_alloc_n; ++index) {
    acc[index] = (pattern == 0)
                     ? (int32_t)((int32_t)19 + (int32_t)(index * 101))
                     : (int32_t)(-137 + (int32_t)n + (int32_t)(index * 29));
    acc_before[index] = acc[index];
  }
  for (size_t index = 0; index < out_alloc_n; ++index) {
    out[index] = -98765.25f;
    if (pattern == 1)
      out[index] = -70000.5f + (float)index;
    old_out[index] = out[index];
  }

  tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(lhs, rhs, acc, scale, out, n);

  int32_t expected_acc = acc[0];
  size_t signed_positive_products = 0;
  size_t signed_negative_products = 0;
  size_t widening_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    expected_acc += product;
    if (product > 0)
      ++signed_positive_products;
    if (product < 0)
      ++signed_negative_products;
    if (product > 127 || product < -128)
      ++widening_products;
  }
  float expected = ((float)expected_acc) * scale;
  float delta = out[0] - expected;
  if (delta < 0.0f)
    delta = -delta;
  if (delta > tolerance) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 mismatch n=%zu pattern=%d scale=%.9g got=%.9g expected=%.9g delta=%.9g tolerance=%.9g acc0=%d expected_acc=%d\n",
            n, pattern, scale, out[0], expected, delta, tolerance, acc[0],
            expected_acc);
    status = 12;
    goto cleanup;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       widening_products == 0)) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 signed widening-product coverage missing n=%zu pattern=%d scale=%.9g positive=%zu negative=%zu widening_products=%zu\n",
            n, pattern, scale, signed_positive_products, signed_negative_products,
            widening_products);
    status = 13;
    goto cleanup;
  }
  for (size_t index = 1; index < out_alloc_n; ++index) {
    if (out[index] != old_out[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 touched tail sentinel n=%zu pattern=%d scale=%.9g index=%zu got=%.9g old=%.9g\n",
              n, pattern, scale, index, out[index], old_out[index]);
      status = 14;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 source buffer mutated n=%zu pattern=%d scale=%.9g index=%zu lhs=%d lhs_before=%d rhs=%d rhs_before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      status = 15;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < acc_alloc_n; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 accumulator buffer mutated n=%zu pattern=%d scale=%.9g index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      status = 16;
      goto cleanup;
    }
  }

  printf("widening_product_reduce_dequantize_f32 case n=%zu pattern=%d scale=%.9g ok expected_acc=%d out0=%.9g tolerance=%.9g signed_positive_products=%zu signed_negative_products=%zu widening_products=%zu source_preserved accumulator_preserved tail_preserved\n",
         n, pattern, scale, expected_acc, out[0], tolerance,
         signed_positive_products, signed_negative_products, widening_products);

cleanup:
  free(lhs);
  free(rhs);
  free(lhs_before);
  free(rhs_before);
  free(acc);
  free(acc_before);
  free(out);
  free(old_out);
  return status;
}

int main(void) {
  const size_t counts[] = {257, 4096, 65536};
  const float scale_values[] = {-0.125f, 0.375f};
  const size_t scale_count = sizeof(scale_values) / sizeof(scale_values[0]);
  if (scale_count < 2) {
    fprintf(stderr, "widening_product_reduce_dequantize_f32 requires at least two runtime scale values\n");
    return 21;
  }
  for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
    if (scale_values[scale_index] == 0.0f) {
      fprintf(stderr, "widening_product_reduce_dequantize_f32 requires nonzero runtime scale values\n");
      return 22;
    }
  }
  for (size_t count_index = 0; count_index < sizeof(counts) / sizeof(counts[0]); ++count_index) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
        int status = run_case(counts[count_index], pattern, scale_values[scale_index]);
        if (status != 0)
          return status;
      }
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_widening_product_reduce_dequantize_f32_ok counts=257,4096,65536 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05\n");
  printf("PASS op=widening_product_reduce_dequantize_f32 counts=257,4096,65536 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05\n");
  return 0;
}
