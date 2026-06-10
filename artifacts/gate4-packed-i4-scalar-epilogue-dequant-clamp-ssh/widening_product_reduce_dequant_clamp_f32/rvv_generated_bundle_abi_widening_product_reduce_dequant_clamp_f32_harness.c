#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

struct BoundPair {
  float lower_bound;
  float upper_bound;
};

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

static float clamp_f32(float value, float lower_bound, float upper_bound) {
  return value < lower_bound ? lower_bound
                             : (value > upper_bound ? upper_bound : value);
}

static int run_case(size_t n, int pattern, float scale,
                    struct BoundPair bounds) {
  /* expected: packed_i4_reference_oracle + dequant scale + clamp */
  const float tolerance = 1e-05f;
  const float lower_bound = bounds.lower_bound;
  const float upper_bound = bounds.upper_bound;
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
            "allocation failed for packed-i4 clamp n=%zu pattern=%d scale=%.9g bounds=[%.9g,%.9g]\n",
            n, pattern, scale, lower_bound, upper_bound);
    status = 11;
    goto cleanup;
  }
  if (scale == 0.0f) {
    fprintf(stderr, "widening_product_reduce_dequant_clamp_f32 requires nonzero runtime scale values\n");
    status = 21;
    goto cleanup;
  }
  if (!(lower_bound < upper_bound)) {
    fprintf(stderr,
            "widening_product_reduce_dequant_clamp_f32 requires ordered runtime bounds lower<upper; lower=%.9g upper=%.9g\n",
            lower_bound, upper_bound);
    status = 22;
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

  tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(lhs, rhs, acc, scale, lower_bound, upper_bound,
                              out, n);

  int32_t expected_acc = acc_before[0];
  size_t signed_positive_products = 0;
  size_t signed_negative_products = 0;
  size_t low_nibble_products = 0;
  size_t high_nibble_products = 0;
  for (size_t index = 0; index < n; ++index) {
    uint8_t lhs_byte = (uint8_t)lhs_before[index];
    uint8_t rhs_byte = (uint8_t)rhs_before[index];
    int32_t lhs_low = sign_extend_i4(lhs_byte & 0x0fu);
    int32_t lhs_high = sign_extend_i4((lhs_byte >> 4) & 0x0fu);
    int32_t rhs_low = sign_extend_i4(rhs_byte & 0x0fu);
    int32_t rhs_high = sign_extend_i4((rhs_byte >> 4) & 0x0fu);
    int32_t low_product = lhs_low * rhs_low;
    int32_t high_product = lhs_high * rhs_high;
    expected_acc += low_product + high_product;
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
  float scaled = ((float)expected_acc) * scale;
  float expected = clamp_f32(scaled, lower_bound, upper_bound);
  float delta = out[0] - expected;
  if (delta < 0.0f)
    delta = -delta;
  if (delta > tolerance) {
    fprintf(stderr,
            "widening_product_reduce_dequant_clamp_f32 packed-i4 clamp mismatch n=%zu pattern=%d scale=%.9g bounds=[%.9g,%.9g] got=%.9g expected=%.9g scaled=%.9g delta=%.9g tolerance=%.9g expected_acc=%d\n",
            n, pattern, scale, lower_bound, upper_bound, out[0], expected,
            scaled, delta, tolerance, expected_acc);
    status = 12;
    goto cleanup;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       low_nibble_products == 0 || high_nibble_products == 0)) {
    fprintf(stderr,
            "widening_product_reduce_dequant_clamp_f32 packed-i4 clamp coverage missing n=%zu pattern=%d scale=%.9g positive=%zu negative=%zu low=%zu high=%zu\n",
            n, pattern, scale, signed_positive_products, signed_negative_products,
            low_nibble_products, high_nibble_products);
    status = 13;
    goto cleanup;
  }
  for (size_t index = 1; index < out_alloc_n; ++index) {
    if (out[index] != old_out[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequant_clamp_f32 packed-i4 clamp touched tail sentinel n=%zu pattern=%d scale=%.9g index=%zu got=%.9g old=%.9g\n",
              n, pattern, scale, index, out[index], old_out[index]);
      status = 14;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequant_clamp_f32 packed-i4 clamp source buffer mutated n=%zu pattern=%d scale=%.9g index=%zu lhs=%d lhs_before=%d rhs=%d rhs_before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      status = 15;
      goto cleanup;
    }
  }
  for (size_t index = 0; index < acc_alloc_n; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequant_clamp_f32 packed-i4 clamp accumulator buffer mutated n=%zu pattern=%d scale=%.9g index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      status = 16;
      goto cleanup;
    }
  }

  printf("widening_product_reduce_dequant_clamp_f32 packed-i4 clamp case n=%zu pattern=%d scale=%.9g bounds=[%.9g,%.9g] ok expected_acc=%d scaled=%.9g out0=%.9g tolerance=%.9g packed_i4_reference_oracle source_preserved accumulator_preserved tail_preserved\n",
         n, pattern, scale, lower_bound, upper_bound, expected_acc, scaled,
         out[0], tolerance);

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
  const struct BoundPair bound_pairs[] = {{-1.5f, 2.25f}, {-8.0f, -0.75f}};
  const size_t scale_count = sizeof(scale_values) / sizeof(scale_values[0]);
  const size_t bound_pair_count = sizeof(bound_pairs) / sizeof(bound_pairs[0]);
  if (scale_count < 2) {
    fprintf(stderr, "widening_product_reduce_dequant_clamp_f32 requires at least two runtime scale values\n");
    return 23;
  }
  if (bound_pair_count < 2) {
    fprintf(stderr, "widening_product_reduce_dequant_clamp_f32 requires at least two runtime bound pairs\n");
    return 24;
  }
  for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
    if (scale_values[scale_index] == 0.0f) {
      fprintf(stderr, "widening_product_reduce_dequant_clamp_f32 requires nonzero runtime scale values\n");
      return 25;
    }
  }
  for (size_t bound_index = 0; bound_index < bound_pair_count; ++bound_index) {
    if (!(bound_pairs[bound_index].lower_bound < bound_pairs[bound_index].upper_bound)) {
      fprintf(stderr,
              "widening_product_reduce_dequant_clamp_f32 requires ordered runtime bound pair at index=%zu lower=%.9g upper=%.9g\n",
              bound_index, bound_pairs[bound_index].lower_bound,
              bound_pairs[bound_index].upper_bound);
      return 26;
    }
  }
  for (size_t count_index = 0; count_index < sizeof(counts) / sizeof(counts[0]); ++count_index) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
        for (size_t bound_index = 0; bound_index < bound_pair_count; ++bound_index) {
          int status = run_case(counts[count_index], pattern,
                                scale_values[scale_index],
                                bound_pairs[bound_index]);
          if (status != 0)
            return status;
        }
      }
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_widening_product_reduce_dequant_clamp_f32_ok counts=257,4096,65536 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05 packed_i4_reference_oracle source_preserved accumulator_preserved tail_preserved\n");
  printf("PASS op=widening_product_reduce_dequant_clamp_f32 counts=257,4096,65536 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05 packed_i4_reference_oracle\n");
  return 0;
}
