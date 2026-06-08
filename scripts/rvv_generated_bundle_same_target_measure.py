#!/usr/bin/env python3
"""Measure generated RVV bundles against scalar C baselines on one RVV target.

This is Gate 4 evidence tooling for the bounded RVV production-kernel
capability campaign. It reuses the generated-bundle ABI e2e path to build and
verify the generated TianChen-RV RVV object/header, then builds a small external
C harness that runs the generated artifact and a named scalar C reference
baseline on the same ``ssh rvv`` target. The script records correctness-before-
timing guards, target profile, compile flags, timing method, raw timing output,
and parsed per-case timing records. It does not implement compiler IR, lowering,
plugin selection, emission, fallback computation, or runtime glue.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shlex
import sys
from string import Template
from typing import Any

import rvv_generated_bundle_abi_e2e as abi


SCRIPT_NAME = "rvv_generated_bundle_same_target_measure"
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/gate4-same-target-measurement")
DEFAULT_OP_KINDS = (
    "widening_product_reduce_dequantize_f32",
    "widening_product_reduce_dequant_clamp_f32",
)
DEFAULT_MEASURE_COUNTS = (257, 4096, 65536)
DEFAULT_WARMUP_COUNT = 2
DEFAULT_REPEAT_COUNT = 5
DEFAULT_MEASURE_ITERATIONS = 8
DEFAULT_COMPILE_FLAGS = ("-O2", "-march=rv64gcv", "-mabi=lp64d", "-I.")
TIMING_METHOD = "clock_gettime(CLOCK_MONOTONIC_RAW)"
BASELINE_IDENTITIES = {
    "widening_product_reduce_dequantize_f32": (
        "scalar-c-reference/product-reduction-dequant-v1"
    ),
    "widening_product_reduce_dequant_clamp_f32": (
        "scalar-c-reference/product-reduction-dequant-clamp-v1"
    ),
}


@dataclass(frozen=True)
class MeasurementConfig:
    counts: list[int]
    dequant_scale_values: list[float]
    warmup_count: int
    repeat_count: int
    measure_iterations: int
    compile_flags: list[str]


def c_string_literal(value: str) -> str:
    return json.dumps(value)


def f32_c_literal(value: float) -> str:
    literal = f"{value:.9g}"
    if "e" not in literal and "E" not in literal and "." not in literal:
        literal += ".0"
    return f"{literal}f"


def measurement_count_summary(counts: list[int]) -> str:
    return ",".join(str(count) for count in counts)


def dequant_scale_summary(values: list[float]) -> str:
    return ",".join(f"{value:.9g}" for value in values)


def default_bound_pairs_literal() -> str:
    return ", ".join(
        f"{{{f32_c_literal(lower)}, {f32_c_literal(upper)}}}"
        for lower, upper in abi.DEFAULT_F32_CLAMP_BOUND_PAIRS
    )


def default_bound_pairs_summary() -> str:
    return ",".join(
        f"{lower:.9g}:{upper:.9g}"
        for lower, upper in abi.DEFAULT_F32_CLAMP_BOUND_PAIRS
    )


def common_measurement_prefix(
    *,
    header_file_name: str,
    expectation: abi.OpExpectation,
    config: MeasurementConfig,
    compile_flags_summary: str,
) -> dict[str, str]:
    return {
        "header_file": header_file_name,
        "op_kind": expectation.kind,
        "function_name": expectation.function_name,
        "baseline_identity": BASELINE_IDENTITIES[expectation.kind],
        "baseline_identity_c": c_string_literal(BASELINE_IDENTITIES[expectation.kind]),
        "generated_identity": (
            f"{expectation.selected_variant}/{expectation.function_name}"
        ),
        "generated_identity_c": c_string_literal(
            f"{expectation.selected_variant}/{expectation.function_name}"
        ),
        "timing_method": TIMING_METHOD,
        "timing_method_c": c_string_literal(TIMING_METHOD),
        "compile_flags_summary": compile_flags_summary,
        "compile_flags_summary_c": c_string_literal(compile_flags_summary),
        "warmup_count": str(config.warmup_count),
        "repeat_count": str(config.repeat_count),
        "measure_iterations": str(config.measure_iterations),
        "counts": ", ".join(str(count) for count in config.counts),
        "counts_summary": measurement_count_summary(config.counts),
        "scales": ", ".join(
            f"{value:.9g}f" for value in config.dequant_scale_values
        ),
        "scales_summary": dequant_scale_summary(config.dequant_scale_values),
        "tolerance": f"{abi.DEQUANT_FLOAT_ABS_TOLERANCE:.9g}f",
        "tolerance_summary": f"{abi.DEQUANT_FLOAT_ABS_TOLERANCE:.9g}",
        "lhs_initializer": expectation.lhs_initializer,
        "rhs_initializer": expectation.rhs_initializer,
        "source_initializer": expectation.source_initializer,
        "dequant_out_sentinel": abi.DEQUANT_F32_OUT_SENTINEL,
        "clamp_out_sentinel": abi.F32_CLAMP_SELECT_OUT_SENTINEL,
        "bound_pairs": default_bound_pairs_literal(),
        "bound_pairs_summary": default_bound_pairs_summary(),
    }


def measurement_harness_source(
    *,
    header_file_name: str,
    expectation: abi.OpExpectation,
    config: MeasurementConfig,
    compile_flags_summary: str,
) -> str:
    values = common_measurement_prefix(
        header_file_name=header_file_name,
        expectation=expectation,
        config=config,
        compile_flags_summary=compile_flags_summary,
    )
    if expectation.is_widening_product_reduce_dequantize_f32:
        return Template(DEQUANT_MEASUREMENT_HARNESS_TEMPLATE).substitute(values)
    if expectation.is_widening_product_reduce_dequant_clamp_f32:
        return Template(CLAMP_MEASUREMENT_HARNESS_TEMPLATE).substitute(values)
    raise abi.EvidenceError(f"unsupported same-target measurement op: {expectation.kind}")


DEQUANT_MEASUREMENT_HARNESS_TEMPLATE = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "$header_file"

#define MEASURE_WARMUPS $warmup_count
#define MEASURE_REPEATS $repeat_count
#define MEASURE_ITERATIONS $measure_iterations

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

__attribute__((noinline)) static void
baseline_product_reduction_dequant_v1(const int8_t *lhs, const int8_t *rhs,
                                      const int32_t *acc, float scale,
                                      float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t index = 0; index < n; ++index)
    sum += (int32_t)lhs[index] * (int32_t)rhs[index];
  out[0] = ((float)sum) * scale;
}

static int init_case(size_t n, int pattern, int8_t *lhs, int8_t *rhs,
                     int8_t *lhs_before, int8_t *rhs_before, int32_t *acc,
                     int32_t *acc_before, float *generated_out,
                     float *baseline_out, float *old_generated_out,
                     float *old_baseline_out) {
  const size_t alloc_n = n == 0 ? 1 : n;
  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = $lhs_initializer;
      rhs[index] = $rhs_initializer;
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
  for (size_t index = 0; index < 4; ++index) {
    acc[index] = (pattern == 0)
                     ? (int32_t)($source_initializer + (int32_t)(index * 101))
                     : (int32_t)(-137 + (int32_t)n + (int32_t)(index * 29));
    acc_before[index] = acc[index];
  }
  for (size_t index = 0; index < 16; ++index) {
    generated_out[index] = $dequant_out_sentinel;
    baseline_out[index] = $dequant_out_sentinel;
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
  const float tolerance = $tolerance;
  baseline_product_reduction_dequant_v1(lhs, rhs, acc, scale, baseline_out, n);
  $function_name(lhs, rhs, acc, scale, generated_out, n);

  int32_t expected_acc = acc_before[0];
  size_t signed_positive_products = 0;
  size_t signed_negative_products = 0;
  size_t widening_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs_before[index] * (int32_t)rhs_before[index];
    expected_acc += product;
    if (product > 0)
      ++signed_positive_products;
    if (product < 0)
      ++signed_negative_products;
    if (product > 127 || product < -128)
      ++widening_products;
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
            "$op_kind correctness mismatch n=%zu pattern=%d scale=%.9g "
            "baseline=%.9g generated=%.9g expected=%.9g "
            "baseline_delta=%.9g generated_delta=%.9g tolerance=%.9g\n",
            n, pattern, scale, baseline_out[0], generated_out[0], expected,
            baseline_delta, generated_delta, tolerance);
    return 12;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       widening_products == 0)) {
    fprintf(stderr,
            "$op_kind signed widening-product coverage missing n=%zu "
            "pattern=%d scale=%.9g positive=%zu negative=%zu widening=%zu\n",
            n, pattern, scale, signed_positive_products,
            signed_negative_products, widening_products);
    return 13;
  }
  for (size_t index = 1; index < 16; ++index) {
    if (generated_out[index] != old_generated_out[index] ||
        baseline_out[index] != old_baseline_out[index]) {
      fprintf(stderr,
              "$op_kind touched tail sentinel n=%zu pattern=%d scale=%.9g "
              "index=%zu generated=%.9g old_generated=%.9g "
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
              "$op_kind source buffer mutated n=%zu pattern=%d scale=%.9g "
              "index=%zu lhs=%d before=%d rhs=%d before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      return 15;
    }
  }
  for (size_t index = 0; index < 4; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "$op_kind accumulator buffer mutated n=%zu pattern=%d scale=%.9g "
              "index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      return 16;
    }
  }
  printf("CORRECTNESS op=$op_kind n=%zu pattern=%d scale=%.9g ok "
         "baseline=$baseline_identity generated=$generated_identity "
         "expected_acc=%d source_preserved accumulator_preserved "
         "tail_preserved\n",
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
    fprintf(stderr, "allocation failed for $op_kind n=%zu\n", n);
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
  printf("CORRECTNESS_GUARD_BEFORE_TIMING op=$op_kind n=%zu pattern=%d "
         "scale=%.9g status=passed\n",
         n, pattern, scale);

  for (int warmup = 0; warmup < MEASURE_WARMUPS; ++warmup) {
    baseline_product_reduction_dequant_v1(lhs, rhs, acc, scale, baseline_out,
                                          n);
    $function_name(lhs, rhs, acc, scale, generated_out, n);
    tcrv_measurement_sink += (double)baseline_out[0] + (double)generated_out[0];
  }

  double best_baseline_per_iter = -1.0;
  double best_generated_per_iter = -1.0;
  for (int repeat = 0; repeat < MEASURE_REPEATS; ++repeat) {
    unsigned long long start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      baseline_product_reduction_dequant_v1(lhs, rhs, acc, scale, baseline_out,
                                            n);
      tcrv_measurement_sink += (double)baseline_out[0];
    }
    unsigned long long baseline_ns = now_ns() - start;

    start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      $function_name(lhs, rhs, acc, scale, generated_out, n);
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
    printf("MEASURE op=$op_kind n=%zu pattern=%d scale=%.9g repeat=%d "
           "baseline_ns=%llu generated_ns=%llu baseline_per_iter_ns=%.3f "
           "generated_per_iter_ns=%.3f speedup=%.6f\n",
           n, pattern, scale, repeat, baseline_ns, generated_ns,
           baseline_per_iter, generated_per_iter, speedup);
  }
  printf("SUMMARY op=$op_kind n=%zu pattern=%d scale=%.9g "
         "baseline_best_per_iter_ns=%.3f generated_best_per_iter_ns=%.3f "
         "best_speedup=%.6f warmups=%d repeats=%d iterations=%d\n",
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
  const size_t counts[] = {$counts};
  const float scale_values[] = {$scales};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scale_count = sizeof(scale_values) / sizeof(scale_values[0]);
  if (scale_count < 2) {
    fprintf(stderr, "$op_kind requires at least two runtime scale values\n");
    return 21;
  }
  printf("MEASURE_CONFIG op=$op_kind baseline=%s generated=%s "
         "target=same-ssh-rvv timing_method=%s compile_flags=%s "
         "counts=$counts_summary scales=$scales_summary warmups=%d repeats=%d "
         "iterations=%d correctness_before_timing=true\n",
         $baseline_identity_c, $generated_identity_c, $timing_method_c,
         $compile_flags_summary_c, MEASURE_WARMUPS, MEASURE_REPEATS,
         MEASURE_ITERATIONS);
  for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
    if (scale_values[scale_index] == 0.0f) {
      fprintf(stderr, "$op_kind requires nonzero runtime scale values\n");
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
  printf("PASS op=$op_kind measurement counts=$counts_summary patterns=0,1 "
         "scales=$scales_summary baseline=$baseline_identity generated=$generated_identity "
         "timing_method=$timing_method warmups=%d repeats=%d iterations=%d "
         "sink=%.9g\n",
         MEASURE_WARMUPS, MEASURE_REPEATS, MEASURE_ITERATIONS,
         tcrv_measurement_sink);
  return 0;
}
'''.lstrip()


CLAMP_MEASUREMENT_HARNESS_TEMPLATE = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "$header_file"

#define MEASURE_WARMUPS $warmup_count
#define MEASURE_REPEATS $repeat_count
#define MEASURE_ITERATIONS $measure_iterations

struct BoundPair {
  float lower_bound;
  float upper_bound;
};

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

__attribute__((noinline)) static void
baseline_product_reduction_dequant_clamp_v1(
    const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale,
    float lower_bound, float upper_bound, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t index = 0; index < n; ++index)
    sum += (int32_t)lhs[index] * (int32_t)rhs[index];
  float scaled = ((float)sum) * scale;
  out[0] = scaled < lower_bound
               ? lower_bound
               : (scaled > upper_bound ? upper_bound : scaled);
}

static int init_case(size_t n, int pattern, int8_t *lhs, int8_t *rhs,
                     int8_t *lhs_before, int8_t *rhs_before, int32_t *acc,
                     int32_t *acc_before, float *generated_out,
                     float *baseline_out, float *old_generated_out,
                     float *old_baseline_out) {
  const size_t alloc_n = n == 0 ? 1 : n;
  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = $lhs_initializer;
      rhs[index] = $rhs_initializer;
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
  for (size_t index = 0; index < 4; ++index) {
    acc[index] = (pattern == 0)
                     ? (int32_t)($source_initializer + (int32_t)(index * 101))
                     : (int32_t)(-137 + (int32_t)n + (int32_t)(index * 29));
    acc_before[index] = acc[index];
  }
  for (size_t index = 0; index < 16; ++index) {
    generated_out[index] = $clamp_out_sentinel;
    baseline_out[index] = $clamp_out_sentinel;
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
                             struct BoundPair bounds, int8_t *lhs,
                             int8_t *rhs, int8_t *lhs_before,
                             int8_t *rhs_before, int32_t *acc,
                             int32_t *acc_before, float *generated_out,
                             float *baseline_out, float *old_generated_out,
                             float *old_baseline_out) {
  const float tolerance = $tolerance;
  const float lower_bound = bounds.lower_bound;
  const float upper_bound = bounds.upper_bound;
  baseline_product_reduction_dequant_clamp_v1(
      lhs, rhs, acc, scale, lower_bound, upper_bound, baseline_out, n);
  $function_name(lhs, rhs, acc, scale, lower_bound, upper_bound, generated_out,
                 n);

  int32_t expected_acc = acc_before[0];
  size_t signed_positive_products = 0;
  size_t signed_negative_products = 0;
  size_t widening_products = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t product = (int32_t)lhs_before[index] * (int32_t)rhs_before[index];
    expected_acc += product;
    if (product > 0)
      ++signed_positive_products;
    if (product < 0)
      ++signed_negative_products;
    if (product > 127 || product < -128)
      ++widening_products;
  }
  float scaled = ((float)expected_acc) * scale;
  float expected =
      scaled < lower_bound ? lower_bound
                           : (scaled > upper_bound ? upper_bound : scaled);
  float baseline_delta = baseline_out[0] - expected;
  float generated_delta = generated_out[0] - expected;
  if (baseline_delta < 0.0f)
    baseline_delta = -baseline_delta;
  if (generated_delta < 0.0f)
    generated_delta = -generated_delta;
  if (baseline_delta > tolerance || generated_delta > tolerance) {
    fprintf(stderr,
            "$op_kind correctness mismatch n=%zu pattern=%d scale=%.9g "
            "bounds=[%.9g,%.9g] baseline=%.9g generated=%.9g "
            "expected=%.9g scaled=%.9g baseline_delta=%.9g "
            "generated_delta=%.9g tolerance=%.9g\n",
            n, pattern, scale, lower_bound, upper_bound, baseline_out[0],
            generated_out[0], expected, scaled, baseline_delta,
            generated_delta, tolerance);
    return 12;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       widening_products == 0)) {
    fprintf(stderr,
            "$op_kind signed widening-product coverage missing n=%zu "
            "pattern=%d scale=%.9g bounds=[%.9g,%.9g] positive=%zu "
            "negative=%zu widening=%zu\n",
            n, pattern, scale, lower_bound, upper_bound,
            signed_positive_products, signed_negative_products,
            widening_products);
    return 13;
  }
  for (size_t index = 1; index < 16; ++index) {
    if (generated_out[index] != old_generated_out[index] ||
        baseline_out[index] != old_baseline_out[index]) {
      fprintf(stderr,
              "$op_kind touched tail sentinel n=%zu pattern=%d scale=%.9g "
              "bounds=[%.9g,%.9g] index=%zu generated=%.9g "
              "old_generated=%.9g baseline=%.9g old_baseline=%.9g\n",
              n, pattern, scale, lower_bound, upper_bound, index,
              generated_out[index], old_generated_out[index],
              baseline_out[index], old_baseline_out[index]);
      return 14;
    }
  }
  const size_t alloc_n = n == 0 ? 1 : n;
  for (size_t index = 0; index < alloc_n; ++index) {
    if (lhs[index] != lhs_before[index] || rhs[index] != rhs_before[index]) {
      fprintf(stderr,
              "$op_kind source buffer mutated n=%zu pattern=%d scale=%.9g "
              "bounds=[%.9g,%.9g] index=%zu lhs=%d before=%d rhs=%d before=%d\n",
              n, pattern, scale, lower_bound, upper_bound, index, lhs[index],
              lhs_before[index], rhs[index], rhs_before[index]);
      return 15;
    }
  }
  for (size_t index = 0; index < 4; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "$op_kind accumulator buffer mutated n=%zu pattern=%d scale=%.9g "
              "bounds=[%.9g,%.9g] index=%zu got=%d before=%d\n",
              n, pattern, scale, lower_bound, upper_bound, index, acc[index],
              acc_before[index]);
      return 16;
    }
  }
  printf("CORRECTNESS op=$op_kind n=%zu pattern=%d scale=%.9g "
         "bounds=[%.9g,%.9g] ok baseline=$baseline_identity "
         "generated=$generated_identity expected_acc=%d source_preserved "
         "accumulator_preserved tail_preserved\n",
         n, pattern, scale, lower_bound, upper_bound, expected_acc);
  return 0;
}

static int run_case(size_t n, int pattern, float scale,
                    struct BoundPair bounds) {
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
    fprintf(stderr, "allocation failed for $op_kind n=%zu\n", n);
    status = 11;
    goto cleanup;
  }

  init_case(n, pattern, lhs, rhs, lhs_before, rhs_before, acc, acc_before,
            generated_out, baseline_out, old_generated_out, old_baseline_out);
  status = correctness_guard(n, pattern, scale, bounds, lhs, rhs, lhs_before,
                             rhs_before, acc, acc_before, generated_out,
                             baseline_out, old_generated_out,
                             old_baseline_out);
  if (status != 0)
    goto cleanup;
  printf("CORRECTNESS_GUARD_BEFORE_TIMING op=$op_kind n=%zu pattern=%d "
         "scale=%.9g bounds=[%.9g,%.9g] status=passed\n",
         n, pattern, scale, bounds.lower_bound, bounds.upper_bound);

  for (int warmup = 0; warmup < MEASURE_WARMUPS; ++warmup) {
    baseline_product_reduction_dequant_clamp_v1(
        lhs, rhs, acc, scale, bounds.lower_bound, bounds.upper_bound,
        baseline_out, n);
    $function_name(lhs, rhs, acc, scale, bounds.lower_bound, bounds.upper_bound,
                   generated_out, n);
    tcrv_measurement_sink += (double)baseline_out[0] + (double)generated_out[0];
  }

  double best_baseline_per_iter = -1.0;
  double best_generated_per_iter = -1.0;
  for (int repeat = 0; repeat < MEASURE_REPEATS; ++repeat) {
    unsigned long long start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      baseline_product_reduction_dequant_clamp_v1(
          lhs, rhs, acc, scale, bounds.lower_bound, bounds.upper_bound,
          baseline_out, n);
      tcrv_measurement_sink += (double)baseline_out[0];
    }
    unsigned long long baseline_ns = now_ns() - start;

    start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      $function_name(lhs, rhs, acc, scale, bounds.lower_bound,
                     bounds.upper_bound, generated_out, n);
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
    printf("MEASURE op=$op_kind n=%zu pattern=%d scale=%.9g "
           "bounds=%.9g:%.9g repeat=%d baseline_ns=%llu "
           "generated_ns=%llu baseline_per_iter_ns=%.3f "
           "generated_per_iter_ns=%.3f speedup=%.6f\n",
           n, pattern, scale, bounds.lower_bound, bounds.upper_bound, repeat,
           baseline_ns, generated_ns, baseline_per_iter, generated_per_iter,
           speedup);
  }
  printf("SUMMARY op=$op_kind n=%zu pattern=%d scale=%.9g bounds=%.9g:%.9g "
         "baseline_best_per_iter_ns=%.3f generated_best_per_iter_ns=%.3f "
         "best_speedup=%.6f warmups=%d repeats=%d iterations=%d\n",
         n, pattern, scale, bounds.lower_bound, bounds.upper_bound,
         best_baseline_per_iter, best_generated_per_iter,
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
  const size_t counts[] = {$counts};
  const float scale_values[] = {$scales};
  const struct BoundPair bound_pairs[] = {$bound_pairs};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scale_count = sizeof(scale_values) / sizeof(scale_values[0]);
  const size_t bound_pair_count = sizeof(bound_pairs) / sizeof(bound_pairs[0]);
  if (scale_count < 2) {
    fprintf(stderr, "$op_kind requires at least two runtime scale values\n");
    return 21;
  }
  if (bound_pair_count < 2) {
    fprintf(stderr, "$op_kind requires at least two runtime bound pairs\n");
    return 22;
  }
  printf("MEASURE_CONFIG op=$op_kind baseline=%s generated=%s "
         "target=same-ssh-rvv timing_method=%s compile_flags=%s "
         "counts=$counts_summary scales=$scales_summary "
         "bound_pairs=$bound_pairs_summary warmups=%d repeats=%d "
         "iterations=%d correctness_before_timing=true\n",
         $baseline_identity_c, $generated_identity_c, $timing_method_c,
         $compile_flags_summary_c, MEASURE_WARMUPS, MEASURE_REPEATS,
         MEASURE_ITERATIONS);
  for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
    if (scale_values[scale_index] == 0.0f) {
      fprintf(stderr, "$op_kind requires nonzero runtime scale values\n");
      return 23;
    }
  }
  for (size_t bound_index = 0; bound_index < bound_pair_count; ++bound_index) {
    if (!(bound_pairs[bound_index].lower_bound <
          bound_pairs[bound_index].upper_bound)) {
      fprintf(stderr,
              "$op_kind requires ordered runtime bound pair at index=%zu "
              "lower=%.9g upper=%.9g\n",
              bound_index, bound_pairs[bound_index].lower_bound,
              bound_pairs[bound_index].upper_bound);
      return 24;
    }
  }
  for (size_t count_index = 0; count_index < count_count; ++count_index) {
    for (int pattern = 0; pattern < 2; ++pattern) {
      for (size_t scale_index = 0; scale_index < scale_count; ++scale_index) {
        for (size_t bound_index = 0; bound_index < bound_pair_count;
             ++bound_index) {
          int status = run_case(counts[count_index], pattern,
                                scale_values[scale_index],
                                bound_pairs[bound_index]);
          if (status != 0)
            return status;
        }
      }
    }
  }
  printf("PASS op=$op_kind measurement counts=$counts_summary patterns=0,1 "
         "scales=$scales_summary bound_pairs=$bound_pairs_summary "
         "baseline=$baseline_identity generated=$generated_identity "
         "timing_method=$timing_method warmups=%d repeats=%d iterations=%d "
         "sink=%.9g\n",
         MEASURE_WARMUPS, MEASURE_REPEATS, MEASURE_ITERATIONS,
         tcrv_measurement_sink);
  return 0;
}
'''.lstrip()


def parse_key_value_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_measurement_stdout(stdout: str) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    correctness: list[dict[str, str]] = []
    configs: list[dict[str, str]] = []
    pass_lines: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("MEASURE_CONFIG "):
            configs.append(parse_key_value_line(line))
        elif line.startswith("CORRECTNESS "):
            correctness.append(parse_key_value_line(line))
        elif line.startswith("MEASURE "):
            records.append(parse_key_value_line(line))
        elif line.startswith("SUMMARY "):
            summaries.append(parse_key_value_line(line))
        elif line.startswith("PASS op="):
            pass_lines.append(line)
    return {
        "config_records": configs,
        "correctness_records": correctness,
        "measurement_records": records,
        "summary_records": summaries,
        "pass_lines": pass_lines,
        "measurement_record_count": len(records),
        "summary_record_count": len(summaries),
        "correctness_record_count": len(correctness),
    }


def make_generation_args(op_kind: str, timeout: int) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=True,
        direct_pre_realized_route_entry=False,
        input=None,
        source_seed=False,
        vector_source_front_door=False,
        pre_realized_selected_body=True,
        rhs_broadcast_selected_body=False,
        lmul_m2_selected_body=False,
        op_kind=[op_kind],
        timeout=timeout,
    )


def selected_pre_realized_expectation(op_kind: str) -> abi.OpExpectation:
    if op_kind not in DEFAULT_OP_KINDS:
        raise abi.EvidenceError(
            f"same-target measurement supports only {', '.join(DEFAULT_OP_KINDS)}"
        )
    expectation = abi.PRE_REALIZED_SELECTED_BODY_OP_EXPECTATIONS[op_kind]
    if expectation.input_path.is_absolute():
        return expectation
    return replace(expectation, input_path=abi.REPO_ROOT / expectation.input_path)


def generate_verified_bundle(
    *,
    args: argparse.Namespace,
    run_id: str,
    artifact_dir: Path,
    expectation: abi.OpExpectation,
    config: MeasurementConfig,
    tcrv_opt: str,
    tcrv_translate: str,
    readobj: str | None,
) -> dict[str, Any]:
    generation_args = make_generation_args(expectation.kind, args.timeout)
    return abi.run_one_op_e2e(
        args=generation_args,
        run_id=run_id,
        artifact_dir=artifact_dir,
        expectation=expectation,
        tcrv_opt=tcrv_opt,
        tcrv_translate=tcrv_translate,
        readobj=readobj,
        runtime_counts=config.counts,
        rhs_scalar_values=list(abi.DEFAULT_RHS_SCALAR_VALUES),
        stride_bytes_values=list(abi.DEFAULT_STRIDED_LOAD_BYTE_STRIDES),
        dequant_scale_values=config.dequant_scale_values,
    )


def run_remote_measurement(
    *,
    op_artifact_dir: Path,
    run_id: str,
    expectation: abi.OpExpectation,
    ssh_target: str,
    connect_timeout: int,
    timeout: int,
    object_path: Path,
    header_path: Path,
    harness_path: Path,
    compile_flags: list[str],
) -> dict[str, Any]:
    remote_dir = (
        f"/tmp/tianchenrv_rvv_same_target_measure_"
        f"{abi.safe_run_id(run_id)}_{expectation.kind}"
    )
    remote_object = f"{remote_dir}/{object_path.name}"
    remote_header = f"{remote_dir}/{header_path.name}"
    remote_harness = f"{remote_dir}/{harness_path.name}"
    remote_binary = f"{remote_dir}/rvv_same_target_measure_{expectation.kind}"

    commands: dict[str, Any] = {"remote_dir": remote_dir}
    setup = abi.run_remote_shell(
        ssh_target,
        connect_timeout,
        f"rm -rf {abi.remote_quote(remote_dir)} && mkdir -p {abi.remote_quote(remote_dir)}",
        timeout,
    )
    commands["setup"] = setup
    abi.require_command_success(setup, "remote measurement setup")

    scp_command = abi.scp_base_command(connect_timeout) + [
        str(object_path),
        str(header_path),
        str(harness_path),
        f"{ssh_target}:{remote_dir}/",
    ]
    scp_record = abi.run_command(scp_command, timeout=timeout)
    commands["scp"] = scp_record
    abi.require_command_success(scp_record, "remote measurement artifact staging")

    profile_command = (
        f"cd {abi.remote_quote(remote_dir)} && "
        f"printf 'ssh_target={ssh_target}\\n' && "
        "printf 'remote_arch=' && uname -m && "
        "printf 'remote_uname=' && uname -a && "
        "printf 'clang_path=' && command -v clang && "
        "printf 'clang_version=' && clang --version | head -n 1 && "
        "printf 'lscpu_begin\\n' && (lscpu | sed -n '1,32p') && "
        "printf 'lscpu_end\\n'"
    )
    profile_record = abi.run_remote_shell(
        ssh_target, connect_timeout, profile_command, timeout, stdout_limit=32768
    )
    commands["target_profile"] = profile_record
    abi.require_command_success(profile_record, "remote target profile capture")

    compile_invocation = shlex.join(
        ["clang"] + compile_flags + [remote_harness, remote_object, "-o", remote_binary]
    )
    compile_command = (
        f"cd {abi.remote_quote(remote_dir)} && "
        f"printf 'compile_flags={shlex.join(compile_flags)}\\n' && "
        f"{compile_invocation}"
    )
    compile_record = abi.run_remote_shell(
        ssh_target, connect_timeout, compile_command, timeout, stdout_limit=32768
    )
    commands["compile"] = compile_record
    abi.require_command_success(compile_record, "remote measurement compile/link")

    run_record = abi.run_remote_shell(
        ssh_target,
        connect_timeout,
        remote_binary,
        timeout,
        stdout_limit=262144,
        stderr_limit=32768,
    )
    commands["run"] = run_record
    abi.require_command_success(run_record, "remote same-target measurement run")
    stdout = str(run_record.get("stdout", ""))
    abi.require_contains(
        stdout,
        f"PASS op={expectation.kind} measurement",
        "remote same-target measurement output",
    )

    cleanup = abi.run_remote_shell(
        ssh_target, connect_timeout, f"rm -rf {abi.remote_quote(remote_dir)}", timeout
    )
    commands["cleanup"] = cleanup
    if cleanup.get("exit_code") != 0:
        commands["cleanup_warning"] = "remote cleanup failed"

    (op_artifact_dir / "remote_target_profile_stdout.txt").write_text(
        str(profile_record.get("stdout", "")), encoding="utf-8"
    )
    (op_artifact_dir / "remote_measure_compile_stdout.txt").write_text(
        str(compile_record.get("stdout", "")), encoding="utf-8"
    )
    (op_artifact_dir / "remote_measure_run_stdout.txt").write_text(
        stdout, encoding="utf-8"
    )
    parsed = parse_measurement_stdout(stdout)
    if not parsed["measurement_records"] or not parsed["summary_records"]:
        raise abi.EvidenceError(
            f"{expectation.kind} measurement output did not contain timing records"
        )
    return {
        "ssh_target": ssh_target,
        "op_kind": expectation.kind,
        "remote_dir": remote_dir,
        "remote_object": remote_object,
        "remote_header": remote_header,
        "remote_harness": remote_harness,
        "remote_binary": remote_binary,
        "remote_compile_succeeded": compile_record.get("exit_code") == 0,
        "remote_run_succeeded": run_record.get("exit_code") == 0,
        "target_profile_stdout": abi.sanitize_text(
            profile_record.get("stdout", ""), limit=32768
        ),
        "remote_output": abi.sanitize_text(stdout, limit=32768),
        "parsed_timing": parsed,
        "commands": commands,
    }


def op_measurement_summary(
    *,
    expectation: abi.OpExpectation,
    generation_result: dict[str, Any],
    harness_path: Path,
    config: MeasurementConfig,
    remote: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle_checks = generation_result["bundle_checks"]
    bundle_dir = Path(generation_result["artifact_dir"]) / "generated_bundle"
    object_path = bundle_dir / bundle_checks["object_file"]
    header_path = bundle_dir / bundle_checks["header_file"]
    summary: dict[str, Any] = {
        "status": "success" if remote else "dry_run_success",
        "op_kind": expectation.kind,
        "baseline_identity": BASELINE_IDENTITIES[expectation.kind],
        "baseline_role": "same-target scalar C comparator and correctness oracle",
        "generated_artifact_identity": {
            "selected_variant": expectation.selected_variant,
            "function_name": expectation.function_name,
            "object": str(object_path),
            "object_sha256": abi.sha256_file(object_path),
            "header": str(header_path),
            "header_sha256": abi.sha256_file(header_path),
            "selected_input": generation_result["selected_input"]["path"],
        },
        "measurement_harness": {
            "path": str(harness_path),
            "sha256": abi.sha256_file(harness_path),
            "timing_method": TIMING_METHOD,
            "correctness_before_timing": True,
            "warmup_count": config.warmup_count,
            "repeat_count": config.repeat_count,
            "measure_iterations": config.measure_iterations,
            "input_sizes": config.counts,
            "dequant_scale_values": config.dequant_scale_values,
            "compile_flags": config.compile_flags,
        },
        "generated_bundle_checks": {
            "status": generation_result["status"],
            "input_mode": generation_result["input_mode"],
            "pre_realized_body_consumed": generation_result.get(
                "materialized_selected_body_checks", {}
            ).get("pre_realized_body_consumed", False),
            "widening_product_reduction_boundary": generation_result.get(
                "widening_product_reduction_boundary", {}
            ),
        },
    }
    if expectation.is_widening_product_reduce_dequant_clamp_f32:
        summary["measurement_harness"]["f32_clamp_bound_pairs"] = [
            {"lower_bound": lower, "upper_bound": upper}
            for lower, upper in abi.DEFAULT_F32_CLAMP_BOUND_PAIRS
        ]
    if remote:
        commands = remote.get("commands", {})
        summary["ssh_measurement_summary"] = {
            "ssh_target": remote.get("ssh_target", ""),
            "remote_dir": remote.get("remote_dir", ""),
            "target_profile_stdout_path": str(
                Path(generation_result["artifact_dir"])
                / "remote_target_profile_stdout.txt"
            ),
            "remote_compile_stdout_path": str(
                Path(generation_result["artifact_dir"])
                / "remote_measure_compile_stdout.txt"
            ),
            "remote_run_stdout_path": str(
                Path(generation_result["artifact_dir"])
                / "remote_measure_run_stdout.txt"
            ),
            "remote_compile_succeeded": remote.get(
                "remote_compile_succeeded", False
            ),
            "remote_run_succeeded": remote.get("remote_run_succeeded", False),
            "target_profile": abi.sanitize_text(
                remote.get("target_profile_stdout", ""), limit=8192
            ),
            "parsed_timing": remote.get("parsed_timing", {}),
            "compile_command": abi.command_summary(commands.get("compile", {})),
            "run_command": abi.command_summary(commands.get("run", {})),
        }
    return summary


def run_one_measurement(
    *,
    args: argparse.Namespace,
    run_id: str,
    artifact_dir: Path,
    expectation: abi.OpExpectation,
    config: MeasurementConfig,
    tcrv_opt: str,
    tcrv_translate: str,
    readobj: str | None,
) -> dict[str, Any]:
    op_artifact_dir = artifact_dir / expectation.kind
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "status": "started",
        "created_at": abi.utc_timestamp(),
        "run_id": run_id,
        "op_kind": expectation.kind,
        "input_mode": "pre-realized-selected-body",
        "dry_run": bool(args.dry_run),
        "ssh_target": args.ssh_target,
        "baseline_identity": BASELINE_IDENTITIES[expectation.kind],
        "timing_method": TIMING_METHOD,
        "measurement_config": {
            "input_sizes": config.counts,
            "dequant_scale_values": config.dequant_scale_values,
            "warmup_count": config.warmup_count,
            "repeat_count": config.repeat_count,
            "measure_iterations": config.measure_iterations,
            "compile_flags": config.compile_flags,
        },
    }
    try:
        generation_result = generate_verified_bundle(
            args=args,
            run_id=run_id,
            artifact_dir=artifact_dir,
            expectation=expectation,
            config=config,
            tcrv_opt=tcrv_opt,
            tcrv_translate=tcrv_translate,
            readobj=readobj,
        )
        evidence["generated_bundle_generation"] = abi.root_op_result_summary(
            expectation=expectation, result=generation_result
        )

        bundle_checks = generation_result["bundle_checks"]
        bundle_dir = op_artifact_dir / "generated_bundle"
        object_path = bundle_dir / bundle_checks["object_file"]
        header_path = bundle_dir / bundle_checks["header_file"]
        harness_path = (
            op_artifact_dir
            / f"rvv_generated_bundle_same_target_measure_{expectation.kind}.c"
        )
        compile_flags_summary = shlex.join(config.compile_flags)
        harness_path.write_text(
            measurement_harness_source(
                header_file_name=bundle_checks["header_file"],
                expectation=expectation,
                config=config,
                compile_flags_summary=compile_flags_summary,
            ),
            encoding="utf-8",
        )
        evidence["measurement_harness"] = {
            "path": str(harness_path),
            "sha256": abi.sha256_file(harness_path),
            "correctness_before_timing": True,
            "baseline_identity": BASELINE_IDENTITIES[expectation.kind],
            "generated_function": expectation.function_name,
        }

        remote: dict[str, Any] | None = None
        if args.dry_run:
            evidence["ssh_evidence"] = False
            evidence["status"] = "dry_run_success"
        else:
            remote = run_remote_measurement(
                op_artifact_dir=op_artifact_dir,
                run_id=run_id,
                expectation=expectation,
                ssh_target=args.ssh_target,
                connect_timeout=args.connect_timeout,
                timeout=args.timeout,
                object_path=object_path,
                header_path=header_path,
                harness_path=harness_path,
                compile_flags=config.compile_flags,
            )
            evidence["remote_measurement"] = remote
            evidence["ssh_evidence"] = True
            evidence["status"] = "success"
        evidence["op_summary"] = op_measurement_summary(
            expectation=expectation,
            generation_result=generation_result,
            harness_path=harness_path,
            config=config,
            remote=remote,
        )
        evidence["completed_at"] = abi.utc_timestamp()
        abi.write_json(op_artifact_dir / "same_target_measurement_evidence.json", evidence)
        return evidence
    except Exception as exc:  # noqa: BLE001 - evidence should record blockers.
        evidence["status"] = "blocked" if not args.dry_run else "failed"
        evidence["completed_at"] = abi.utc_timestamp()
        evidence["diagnostic"] = abi.sanitize_text(exc)
        op_artifact_dir.mkdir(parents=True, exist_ok=True)
        abi.write_json(op_artifact_dir / "same_target_measurement_evidence.json", evidence)
        raise abi.EvidenceError(
            f"{expectation.kind} same-target measurement failed: {exc}"
        ) from exc


def validate_measurement_config(config: MeasurementConfig) -> None:
    if any(count <= 0 for count in config.counts):
        raise abi.EvidenceError(
            f"same-target measurement counts must be positive: {config.counts}"
        )
    abi.validate_runtime_counts(config.counts)
    abi.validate_dequant_scale_values(config.dequant_scale_values)
    if config.warmup_count < 0:
        raise abi.EvidenceError("--warmup-count must be >= 0")
    if config.repeat_count < 1:
        raise abi.EvidenceError("--repeat-count must be >= 1")
    if config.measure_iterations < 1:
        raise abi.EvidenceError("--measure-iterations must be >= 1")
    if not config.compile_flags:
        raise abi.EvidenceError("at least one compile flag must be recorded")


def run_measurement(args: argparse.Namespace) -> int:
    run_id = abi.safe_run_id(args.run_id or abi.utc_run_id())
    artifact_dir = abi.prepare_artifact_dir(args.artifact_root, run_id, args.overwrite)
    config = MeasurementConfig(
        counts=args.measure_count or list(DEFAULT_MEASURE_COUNTS),
        dequant_scale_values=args.dequant_scale or list(abi.DEFAULT_DEQUANT_SCALE_VALUES),
        warmup_count=args.warmup_count,
        repeat_count=args.repeat_count,
        measure_iterations=args.measure_iterations,
        compile_flags=args.compile_flag or list(DEFAULT_COMPILE_FLAGS),
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "status": "started",
        "created_at": abi.utc_timestamp(),
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "artifact_dir": str(artifact_dir),
        "input_mode": "pre-realized-selected-body",
        "ssh_target": args.ssh_target,
        "same_target_measurement": True,
        "timing_method": TIMING_METHOD,
        "measurement_config": {
            "input_sizes": config.counts,
            "dequant_scale_values": config.dequant_scale_values,
            "warmup_count": config.warmup_count,
            "repeat_count": config.repeat_count,
            "measure_iterations": config.measure_iterations,
            "compile_flags": config.compile_flags,
            "baseline_identities": BASELINE_IDENTITIES,
        },
        "op_results": {},
    }
    try:
        validate_measurement_config(config)
        op_kinds = args.op_kind or list(DEFAULT_OP_KINDS)
        if len(set(op_kinds)) != len(op_kinds):
            raise abi.EvidenceError(f"duplicate --op-kind values are not allowed: {op_kinds}")
        expectations = [selected_pre_realized_expectation(op_kind) for op_kind in op_kinds]
        evidence["op_kinds"] = [expectation.kind for expectation in expectations]

        tcrv_opt = abi.ensure_tool(args.tcrv_opt)
        tcrv_translate = abi.ensure_tool(args.tcrv_translate)
        readobj = abi.ensure_tool(args.llvm_readobj) if args.llvm_readobj else None

        for expectation in expectations:
            result = run_one_measurement(
                args=args,
                run_id=run_id,
                artifact_dir=artifact_dir,
                expectation=expectation,
                config=config,
                tcrv_opt=tcrv_opt,
                tcrv_translate=tcrv_translate,
                readobj=readobj,
            )
            evidence["op_results"][expectation.kind] = result["op_summary"]

        evidence["ssh_evidence"] = not args.dry_run
        evidence["status"] = "success" if not args.dry_run else "dry_run_success"
        evidence["completed_at"] = abi.utc_timestamp()
        abi.write_json(artifact_dir / "evidence.json", evidence)
        print(f"{SCRIPT_NAME}: {evidence['status']}")
        print(f"artifact_dir: {artifact_dir}")
        if evidence["ssh_evidence"]:
            for op_kind, result in evidence["op_results"].items():
                parsed = (
                    result.get("ssh_measurement_summary", {})
                    .get("parsed_timing", {})
                )
                print(
                    f"[{op_kind}] summaries={parsed.get('summary_record_count', 0)} "
                    f"measurements={parsed.get('measurement_record_count', 0)}"
                )
        return 0
    except Exception as exc:  # noqa: BLE001 - evidence should record blockers.
        evidence["status"] = "blocked" if not args.dry_run else "failed"
        evidence["completed_at"] = abi.utc_timestamp()
        evidence["diagnostic"] = abi.sanitize_text(exc)
        abi.write_json(artifact_dir / "evidence.json", evidence)
        print(f"{SCRIPT_NAME}: {evidence['status']}", file=sys.stderr)
        print(evidence["diagnostic"], file=sys.stderr)
        print(f"artifact_dir: {artifact_dir}", file=sys.stderr)
        return 1


def run_self_test() -> int:
    config = MeasurementConfig(
        counts=[257, 1024],
        dequant_scale_values=list(abi.DEFAULT_DEQUANT_SCALE_VALUES),
        warmup_count=1,
        repeat_count=2,
        measure_iterations=3,
        compile_flags=list(DEFAULT_COMPILE_FLAGS),
    )
    validate_measurement_config(config)
    for op_kind in DEFAULT_OP_KINDS:
        expectation = selected_pre_realized_expectation(op_kind)
        harness = measurement_harness_source(
            header_file_name="artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h",
            expectation=expectation,
            config=config,
            compile_flags_summary=shlex.join(config.compile_flags),
        )
        required_tokens = [
            BASELINE_IDENTITIES[op_kind],
            expectation.function_name,
            "CLOCK_MONOTONIC_RAW",
            "CORRECTNESS_GUARD_BEFORE_TIMING",
            "MEASURE_CONFIG op=" + op_kind,
            "MEASURE op=" + op_kind,
            "SUMMARY op=" + op_kind,
            "baseline_ns",
            "generated_ns",
            "MEASURE_ITERATIONS",
            "PASS op=" + op_kind + " measurement",
        ]
        for token in required_tokens:
            if token not in harness:
                raise AssertionError(
                    f"self-test harness for {op_kind} lost token: {token}"
                )
    parsed = parse_measurement_stdout(
        "\n".join(
            [
                "MEASURE_CONFIG op=widening_product_reduce_dequantize_f32",
                "CORRECTNESS op=widening_product_reduce_dequantize_f32 n=257",
                "MEASURE op=widening_product_reduce_dequantize_f32 n=257 "
                "baseline_ns=100 generated_ns=50",
                "SUMMARY op=widening_product_reduce_dequantize_f32 n=257 "
                "baseline_best_per_iter_ns=100 generated_best_per_iter_ns=50",
                "PASS op=widening_product_reduce_dequantize_f32 measurement",
            ]
        )
    )
    if (
        parsed["measurement_record_count"] != 1
        or parsed["summary_record_count"] != 1
        or parsed["correctness_record_count"] != 1
        or not parsed["pass_lines"]
    ):
        raise AssertionError("self-test measurement stdout parser lost records")
    print(f"{SCRIPT_NAME} self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate and verify local bundles and measurement harnesses without ssh rvv",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--op-kind",
        choices=DEFAULT_OP_KINDS,
        action="append",
        default=[],
        help="bounded Gate 4 op kind to measure; may be repeated",
    )
    parser.add_argument(
        "--measure-count",
        action="append",
        type=int,
        default=[],
        help="positive runtime n/input size to measure; may be repeated",
    )
    parser.add_argument(
        "--dequant-scale",
        action="append",
        type=float,
        default=[],
        help="runtime f32 scale value to use in correctness and timing cases",
    )
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument(
        "--measure-iterations", type=int, default=DEFAULT_MEASURE_ITERATIONS
    )
    parser.add_argument(
        "--compile-flag",
        action="append",
        default=[],
        help=(
            "remote clang compile flag to record and use; may be repeated; "
            "defaults to the bounded RVV clang flags"
        ),
    )
    parser.add_argument("--tcrv-opt", default="build/bin/tcrv-opt")
    parser.add_argument("--tcrv-translate", default="build/bin/tcrv-translate")
    parser.add_argument("--llvm-readobj", default=abi.default_readobj())
    parser.add_argument("--ssh-target", default=abi.DEFAULT_SSH_TARGET)
    parser.add_argument("--timeout", type=int, default=abi.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=abi.DEFAULT_CONNECT_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    return run_measurement(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
