#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

#define MEASURE_WARMUPS 2
#define MEASURE_REPEATS 5
#define MEASURE_ITERATIONS 8

static volatile double tcrv_measurement_sink = 0.0;

static unsigned long long now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
    exit(97);
  }
  return (unsigned long long)ts.tv_sec * 1000000000ULL +
         (unsigned long long)ts.tv_nsec;
}

static int32_t sign_extend_i4(uint8_t nibble) {
  int32_t value = (int32_t)(nibble & 0x0fu);
  return value >= 8 ? value - 16 : value;
}

static uint8_t encode_signed_i4(int32_t value) {
  return (uint8_t)value & 0x0fu;
}

static int8_t pack_signed_i4_pair(int32_t low, int32_t high) {
  return (int8_t)(encode_signed_i4(low) |
                  (uint8_t)(encode_signed_i4(high) << 4));
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

__attribute__((noinline)) static void
baseline_product_reduction_dequant_packed_i4_v1(
    const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale,
    float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t index = 0; index < n; ++index) {
    uint8_t lhs_byte = (uint8_t)lhs[index];
    uint8_t rhs_byte = (uint8_t)rhs[index];
    int32_t lhs_low = sign_extend_i4(lhs_byte & 0x0fu);
    int32_t lhs_high = sign_extend_i4((lhs_byte >> 4) & 0x0fu);
    int32_t rhs_low = sign_extend_i4(rhs_byte & 0x0fu);
    int32_t rhs_high = sign_extend_i4((rhs_byte >> 4) & 0x0fu);
    sum += lhs_low * rhs_low;
    sum += lhs_high * rhs_high;
  }
  out[0] = ((float)sum) * scale;
}

static int init_case(size_t n, int pattern, int8_t *lhs, int8_t *rhs,
                     int8_t *lhs_before, int8_t *rhs_before, int32_t *acc,
                     int32_t *acc_before, float *generated_out,
                     float *baseline_out, float *old_generated_out,
                     float *old_baseline_out) {
  const size_t alloc_n = n == 0 ? 1 : n;
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
  for (size_t index = 0; index < 4; ++index) {
    acc[index] = (pattern == 0)
                     ? (int32_t)((int32_t)19 + (int32_t)(index * 101))
                     : (int32_t)(-137 + (int32_t)n + (int32_t)(index * 29));
    acc_before[index] = acc[index];
  }
  for (size_t index = 0; index < 16; ++index) {
    generated_out[index] = -98765.25f;
    baseline_out[index] = -98765.25f;
    if (pattern == 1) {
      generated_out[index] = -70000.5f + (float)index;
      baseline_out[index] = -71000.5f + (float)index;
    }
    old_generated_out[index] = generated_out[index];
    old_baseline_out[index] = baseline_out[index];
  }
  return 0;
}

static int correctness_guard(size_t n, int pattern, float scale,
                             int8_t *lhs, int8_t *rhs, int8_t *lhs_before,
                             int8_t *rhs_before, int32_t *acc,
                             int32_t *acc_before, float *generated_out,
                             float *baseline_out, float *old_generated_out,
                             float *old_baseline_out) {
  const float tolerance = 1e-05f;
  baseline_product_reduction_dequant_packed_i4_v1(lhs, rhs, acc, scale,
                                                  baseline_out, n);
  tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(lhs, rhs, acc, scale, generated_out, n);

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
  float expected = ((float)expected_acc) * scale;
  float baseline_delta = baseline_out[0] - expected;
  float generated_delta = generated_out[0] - expected;
  if (baseline_delta < 0.0f)
    baseline_delta = -baseline_delta;
  if (generated_delta < 0.0f)
    generated_delta = -generated_delta;
  if (baseline_delta > tolerance || generated_delta > tolerance) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 packed-i4 correctness mismatch n=%zu pattern=%d "
            "scale=%.9g baseline=%.9g generated=%.9g expected=%.9g "
            "baseline_delta=%.9g generated_delta=%.9g tolerance=%.9g "
            "expected_acc=%d\n",
            n, pattern, scale, baseline_out[0], generated_out[0], expected,
            baseline_delta, generated_delta, tolerance, expected_acc);
    return 12;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       low_nibble_products == 0 || high_nibble_products == 0)) {
    fprintf(stderr,
            "widening_product_reduce_dequantize_f32 packed-i4 coverage missing n=%zu pattern=%d scale=%.9g "
            "positive=%zu negative=%zu low=%zu high=%zu\n",
            n, pattern, scale, signed_positive_products,
            signed_negative_products, low_nibble_products,
            high_nibble_products);
    return 13;
  }
  for (size_t index = 1; index < 16; ++index) {
    if (generated_out[index] != old_generated_out[index] ||
        baseline_out[index] != old_baseline_out[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 touched tail sentinel n=%zu pattern=%d "
              "scale=%.9g index=%zu generated=%.9g old_generated=%.9g "
              "baseline=%.9g old_baseline=%.9g\n",
              n, pattern, scale, index, generated_out[index],
              old_generated_out[index], baseline_out[index],
              old_baseline_out[index]);
      return 14;
    }
  }
  const size_t alloc_n = n == 0 ? 1 : n;
  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 source buffer mutated n=%zu pattern=%d "
              "scale=%.9g index=%zu lhs=%d before=%d rhs=%d before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      return 15;
    }
  }
  for (size_t index = 0; index < 4; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "widening_product_reduce_dequantize_f32 packed-i4 accumulator buffer mutated n=%zu "
              "pattern=%d scale=%.9g index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      return 16;
    }
  }
  printf("CORRECTNESS op=widening_product_reduce_dequantize_f32 n=%zu pattern=%d scale=%.9g ok "
         "baseline=scalar-c-reference/product-reduction-dequant-packed-i4-v1 generated=pre_realized_body_rvv_product_reduce_dequantize/tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize "
         "expected_acc=%d packed_i4_reference_oracle source_preserved "
         "accumulator_preserved tail_preserved\n",
         n, pattern, scale, expected_acc);
  return 0;
}

static int run_case(size_t n, int pattern, float scale) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int8_t *lhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *lhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int8_t *rhs_before = (int8_t *)malloc(sizeof(int8_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  int32_t *acc_before = (int32_t *)malloc(sizeof(int32_t) * 4);
  float *generated_out = (float *)malloc(sizeof(float) * 16);
  float *baseline_out = (float *)malloc(sizeof(float) * 16);
  float *old_generated_out = (float *)malloc(sizeof(float) * 16);
  float *old_baseline_out = (float *)malloc(sizeof(float) * 16);
  int status = 0;
  if (!lhs || !rhs || !lhs_before || !rhs_before || !acc || !acc_before ||
      !generated_out || !baseline_out || !old_generated_out ||
      !old_baseline_out) {
    fprintf(stderr, "allocation failed for widening_product_reduce_dequantize_f32 packed-i4 n=%zu\n", n);
    status = 11;
    goto cleanup;
  }

  init_case(n, pattern, lhs, rhs, lhs_before, rhs_before, acc, acc_before,
            generated_out, baseline_out, old_generated_out, old_baseline_out);
  status = correctness_guard(n, pattern, scale, lhs, rhs, lhs_before,
                             rhs_before, acc, acc_before, generated_out,
                             baseline_out, old_generated_out,
                             old_baseline_out);
  if (status != 0)
    goto cleanup;
  printf("CORRECTNESS_GUARD_BEFORE_TIMING op=widening_product_reduce_dequantize_f32 n=%zu pattern=%d "
         "scale=%.9g status=passed packed_i4_reference_oracle\n",
         n, pattern, scale);

  for (int warmup = 0; warmup < MEASURE_WARMUPS; ++warmup) {
    baseline_product_reduction_dequant_packed_i4_v1(lhs, rhs, acc, scale,
                                                    baseline_out, n);
    tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(lhs, rhs, acc, scale, generated_out, n);
    tcrv_measurement_sink += (double)baseline_out[0] + (double)generated_out[0];
  }

  double best_baseline_per_iter = -1.0;
  double best_generated_per_iter = -1.0;
  for (int repeat = 0; repeat < MEASURE_REPEATS; ++repeat) {
    unsigned long long start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      baseline_product_reduction_dequant_packed_i4_v1(lhs, rhs, acc, scale,
                                                      baseline_out, n);
      tcrv_measurement_sink += (double)baseline_out[0];
    }
    unsigned long long baseline_ns = now_ns() - start;

    start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(lhs, rhs, acc, scale, generated_out, n);
      tcrv_measurement_sink += (double)generated_out[0];
    }
    unsigned long long generated_ns = now_ns() - start;

    double baseline_per_iter =
        (double)baseline_ns / (double)MEASURE_ITERATIONS;
    double generated_per_iter =
        (double)generated_ns / (double)MEASURE_ITERATIONS;
    if (best_baseline_per_iter < 0.0 ||
        baseline_per_iter < best_baseline_per_iter)
      best_baseline_per_iter = baseline_per_iter;
    if (best_generated_per_iter < 0.0 ||
        generated_per_iter < best_generated_per_iter)
      best_generated_per_iter = generated_per_iter;
    double speedup =
        generated_per_iter > 0.0 ? baseline_per_iter / generated_per_iter : 0.0;
    printf("MEASURE op=widening_product_reduce_dequantize_f32 n=%zu pattern=%d scale=%.9g repeat=%d "
           "baseline_ns=%llu generated_ns=%llu baseline_per_iter_ns=%.3f "
           "generated_per_iter_ns=%.3f speedup=%.6f "
           "packed_i4_reference_oracle\n",
           n, pattern, scale, repeat, baseline_ns, generated_ns,
           baseline_per_iter, generated_per_iter, speedup);
  }
  printf("SUMMARY op=widening_product_reduce_dequantize_f32 n=%zu pattern=%d scale=%.9g "
         "baseline_best_per_iter_ns=%.3f generated_best_per_iter_ns=%.3f "
         "best_speedup=%.6f warmups=%d repeats=%d iterations=%d "
         "packed_i4_reference_oracle\n",
         n, pattern, scale, best_baseline_per_iter, best_generated_per_iter,
         best_generated_per_iter > 0.0
             ? best_baseline_per_iter / best_generated_per_iter
             : 0.0,
         MEASURE_WARMUPS, MEASURE_REPEATS, MEASURE_ITERATIONS);

cleanup:
  free(lhs);
  free(rhs);
  free(lhs_before);
  free(rhs_before);
  free(acc);
  free(acc_before);
  free(generated_out);
  free(baseline_out);
  free(old_generated_out);
  free(old_baseline_out);
  return status;
}

int main(void) {
  const size_t counts[] = {257, 4096, 65536};
  const float scale_values[] = {-0.125f, 0.375f};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scale_count = sizeof(scale_values) / sizeof(scale_values[0]);
  if (scale_count < 2) {
    fprintf(stderr, "widening_product_reduce_dequantize_f32 requires at least two runtime scale values\n");
    return 21;
  }
  printf("MEASURE_CONFIG op=widening_product_reduce_dequantize_f32 baseline=%s generated=%s "
         "target=same-ssh-rvv timing_method=%s compile_flags=%s "
         "counts=257,4096,65536 scales=-0.125,0.375 warmups=%d repeats=%d "
         "iterations=%d correctness_before_timing=true "
         "packed_i4_reference_oracle=true runtime_n_unit=packed_bytes\n",
         "scalar-c-reference/product-reduction-dequant-packed-i4-v1", "pre_realized_body_rvv_product_reduce_dequantize/tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize", "clock_gettime(CLOCK_MONOTONIC_RAW)",
         "-O2 -march=rv64gcv -mabi=lp64d -I.", MEASURE_WARMUPS, MEASURE_REPEATS,
         MEASURE_ITERATIONS);
  for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
    if (scale_values[scale_index] == 0.0f) {
      fprintf(stderr, "widening_product_reduce_dequantize_f32 requires nonzero runtime scale values\n");
      return 22;
    }
  }
  for (size_t count_index = 0; count_index < count_count; ++count_index) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
        int status =
            run_case(counts[count_index], pattern, scale_values[scale_index]);
        if (status != 0)
          return status;
      }
    }
  }
  printf("PASS op=widening_product_reduce_dequantize_f32 measurement counts=257,4096,65536 patterns=0,1 "
         "scales=-0.125,0.375 baseline=scalar-c-reference/product-reduction-dequant-packed-i4-v1 "
         "generated=pre_realized_body_rvv_product_reduce_dequantize/tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize timing_method=clock_gettime(CLOCK_MONOTONIC_RAW) "
         "warmups=%d repeats=%d iterations=%d packed_i4_reference_oracle "
         "sink=%.9g\n",
         MEASURE_WARMUPS, MEASURE_REPEATS, MEASURE_ITERATIONS,
         tcrv_measurement_sink);
  return 0;
}
