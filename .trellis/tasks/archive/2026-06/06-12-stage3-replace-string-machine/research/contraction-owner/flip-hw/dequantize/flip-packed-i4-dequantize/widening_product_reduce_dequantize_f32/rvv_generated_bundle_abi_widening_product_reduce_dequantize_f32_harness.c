#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int32_t sign_extend_i4(uint8_t nibble) {
  int32_t value = (int32_t)(nibble & 0x0fu);
  return value >= 8 ? value - 16 : value;
}

static uint8_t encode_signed_i4(int32_t value) {
  return (uint8_t)value & 0x0fu;
}

static int8_t pack_signed_i4_pair(int32_t low, int32_t high) {
  return (int8_t)(encode_signed_i4(low) | (uint8_t)(encode_signed_i4(high) << 4));
}

static int32_t lhs_low_value(size_t index, int pattern) {
  static const int32_t pattern0[] = {-8, -3, 2, 7, -1, 5, -6, 4};
  static const int32_t pattern1[] = {6, -7, -2, 3, 1, -5, 7, -4};
  const int32_t *values = pattern == 0 ? pattern0 : pattern1;
  return values[index % 8];
}

static int32_t lhs_high_value(size_t index, int pattern) {
  static const int32_t pattern0[] = {5, -4, 7, -8, 1, -2, 6, -6};
  static const int32_t pattern1[] = {-3, 4, -8, 2, -6, 7, -1, 5};
  const int32_t *values = pattern == 0 ? pattern0 : pattern1;
  return values[(index + 3) % 8];
}

static int32_t rhs_low_value(size_t index, int pattern) {
  static const int32_t pattern0[] = {3, -6, 4, -2, 7, -8, 1, -5};
  static const int32_t pattern1[] = {-4, 7, -1, 6, -8, 2, 5, -3};
  const int32_t *values = pattern == 0 ? pattern0 : pattern1;
  return values[(index + 1) % 8];
}

static int32_t rhs_high_value(size_t index, int pattern) {
  static const int32_t pattern0[] = {-7, 2, -5, 6, -3, 4, -1, 7};
  static const int32_t pattern1[] = {5, -2, 7, -6, 3, -8, 4, -1};
  const int32_t *values = pattern == 0 ? pattern0 : pattern1;
  return values[(index + 5) % 8];
}

static int run_case(size_t n, int pattern, float scale) {
  /* expected: ((float)(acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i]))) * scale */
  /* packed_i4_reference: acc[0] + sum_bytes(lhs.low_i4 * rhs.low_i4 + lhs.high_i4 * rhs.high_i4), then f32 scale */
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
            "allocation failed for packed-i4 n=%zu pattern=%d scale=%.9g\n",
            n, pattern, scale);
    status = 11;
    goto cleanup;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    int32_t lhs_low = lhs_low_value(index, pattern);
    int32_t lhs_high = lhs_high_value(index, pattern);
    int32_t rhs_low = rhs_low_value(index, pattern);
    int32_t rhs_high = rhs_high_value(index, pattern);
    lhs[index] = pack_signed_i4_pair(lhs_low, lhs_high);
    rhs[index] = pack_signed_i4_pair(rhs_low, rhs_high);
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
  size_t low_nibble_products = 0;
  size_t high_nibble_products = 0;
  size_t packed_bytes_checked = 0;
  for (size_t index = 0; index < n; ++index) {
    uint8_t lhs_byte = (uint8_t)lhs[index];
    uint8_t rhs_byte = (uint8_t)rhs[index];
    int32_t lhs_low = sign_extend_i4(lhs_byte & 0x0fu);
    int32_t lhs_high = sign_extend_i4((lhs_byte >> 4) & 0x0fu);
    int32_t rhs_low = sign_extend_i4(rhs_byte & 0x0fu);
    int32_t rhs_high = sign_extend_i4((rhs_byte >> 4) & 0x0fu);
    int32_t low_product = lhs_low * rhs_low;
    int32_t high_product = lhs_high * rhs_high;
    expected_acc += low_product + high_product;
    ++packed_bytes_checked;
    if (low_product != 0)
      ++low_nibble_products;
    if (high_product != 0)
      ++high_nibble_products;
    if (low_product > 0)
      ++signed_positive_products;
    if (low_product < 0)
      ++signed_negative_products;
    if (high_product > 0)
      ++signed_positive_products;
    if (high_product < 0)
      ++signed_negative_products;
  }
  float expected = ((float)expected_acc) * scale;
  float delta = out[0] - expected;
  if (delta < 0.0f)
    delta = -delta;
  if (delta > tolerance) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 packed-i4 mismatch n=%zu pattern=%d scale=%.9g got=%.9g expected=%.9g delta=%.9g tolerance=%.9g acc0=%d expected_acc=%d\n",
            n, pattern, scale, out[0], expected, delta, tolerance, acc[0],
            expected_acc);
    status = 12;
    goto cleanup;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       low_nibble_products == 0 || high_nibble_products == 0)) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 packed-i4 coverage missing n=%zu pattern=%d scale=%.9g positive=%zu negative=%zu low=%zu high=%zu\n",
            n, pattern, scale, signed_positive_products, signed_negative_products,
            low_nibble_products, high_nibble_products);
    status = 13;
    goto cleanup;
  }
  for (size_t index = 1; index < out_alloc_n; ++index) {
    if (out[index] != old_out[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 touched tail sentinel n=%zu pattern=%d scale=%.9g index=%zu got=%.9g old=%.9g\n",
              n, pattern, scale, index, out[index], old_out[index]);
      status = 14;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 source buffer mutated n=%zu pattern=%d scale=%.9g index=%zu lhs=%d lhs_before=%d rhs=%d rhs_before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      status = 15;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < acc_alloc_n; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 accumulator buffer mutated n=%zu pattern=%d scale=%.9g index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      status = 16;
      goto cleanup;
    }
  }

  printf("widening_product_reduce_dequantize_f32 packed-i4 case n=%zu pattern=%d scale=%.9g ok expected_acc=%d out0=%.9g tolerance=%.9g packed_bytes=%zu signed_positive_products=%zu signed_negative_products=%zu low_nibble_products=%zu high_nibble_products=%zu source_preserved accumulator_preserved tail_preserved\n",
         n, pattern, scale, expected_acc, out[0], tolerance,
         packed_bytes_checked, signed_positive_products, signed_negative_products,
         low_nibble_products, high_nibble_products);

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
  const size_t counts[] = {0, 1, 16, 17, 257};
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
  printf("tcrv_rvv_generated_bundle_abi_widening_product_reduce_dequantize_f32_ok counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05 packed_i4_reference_oracle\n");
  printf("PASS op=widening_product_reduce_dequantize_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05 packed_i4_reference_oracle\n");
  return 0;
}
