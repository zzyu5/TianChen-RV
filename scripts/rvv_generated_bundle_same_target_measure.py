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
PACKED_I4_BASELINE_IDENTITIES = {
    "widening_product_reduce_dequantize_f32": (
        "scalar-c-reference/product-reduction-dequant-packed-i4-v1"
    ),
    "widening_product_reduce_dequant_clamp_f32": (
        "scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1"
    ),
}
RESULT_CLASSIFICATION_NOT_MEASURED = "not-measured"
RESULT_CLASSIFICATION_WIN = "win"
RESULT_CLASSIFICATION_NO_WIN = "no-win"
RESULT_CLASSIFICATION_REGRESSION = "regression"
PACKED_I4_MATURITY_CONTRACT_EVIDENCE_INPUT = (
    "packed-i4-same-target-performance-maturity-evidence-input.v1"
)
PACKED_I4_MATURITY_CONTRACT_AUTHORITY = (
    "measurement-evidence-input-only; provider-owned "
    "low-precision resource facts and target artifact mirrors remain "
    "the maturity contract"
)
SOURCE_BACKED_MEASUREMENT_RECORD_CONTRACT = (
    "rvv-low-precision-source-backed-artifact-measurement-record.v1"
)
GENERATED_ARTIFACT_IDENTITY_CONTRACT = (
    "generated-object-header-sha256-after-target-artifact-validation.v1"
)
MEASUREMENT_TARGET_PROVENANCE = "same-target-measurement-workflow-ssh-target.v1"
MEASUREMENT_RUNTIME_COUNT_PROVENANCE = (
    "same-target-measurement-config-input-sizes.v1"
)
PRODUCTION_PRESSURE_PROFILE_LABEL = (
    "low-precision-quantized-contraction-production-pressure"
)
PRODUCTION_PRESSURE_PROFILE_LABEL_PROVENANCE = (
    "non-authoritative-pressure-label-derived-from-selected-typed-rvv-"
    "provider-facts-and-source-backed-measurement-record"
)
PACKED_I4_SAME_TARGET_MEASUREMENT_RECORD_FIELDS = (
    "contract",
    "authority",
    "measurement_evidence_id",
    "measurement_classification",
    "measurement_outcome_family",
    "measurement_best_speedup_range",
    "measurement_summary_record_count",
    "measurement_record_count",
    "correctness_record_count",
    "same_target_measurement",
    "ssh_evidence",
    "target_profile",
    "source_record_contract",
    "source_selected_variant",
    "source_selected_input",
    "source_generated_function",
    "generated_artifact_identity_contract",
    "generated_artifact_object_path",
    "generated_artifact_object_sha256",
    "generated_artifact_header_path",
    "generated_artifact_header_sha256",
    "measurement_target",
    "measurement_target_provenance",
    "measurement_runtime_count_set",
    "measurement_runtime_count_provenance",
    "pressure_profile_label",
    "pressure_profile_label_provenance",
    "provider_resource_selected_candidate",
    "provider_resource_planning_contract",
    "provider_resource_operand_form",
    "provider_resource_source_signedness",
    "provider_resource_storage_element_width",
    "provider_resource_effective_element_width",
    "provider_resource_packing_layout",
    "provider_resource_unpack_intent",
    "provider_resource_vsetvl_region_count",
    "provider_runtime_avl_source",
    "provider_resource_route_family_plan",
    "provider_supported_mirror",
    "provider_runtime_abi_order",
    "provider_schedule_decision_contract",
    "provider_schedule_decision",
    "provider_schedule_decision_reason",
    "provider_resource_cost_contract",
    "provider_resource_cost_model",
    "provider_resource_cost_loop_body_steps",
    "provider_resource_cost_blocker",
    "provider_performance_admission_decision",
    "provider_realization_admission_contract",
    "provider_realization_admission_decision",
    "provider_realization_admission_evidence",
    "provider_realization_admission_dispatch_policy",
    "provider_realization_admission_schedule_decision_contract",
    "provider_realization_admission_schedule_decision",
    "provider_realization_admission_schedule_decision_reason",
    "provider_primitive_chain_contract",
    "provider_primitive_chain_kind",
    "provider_primitive_contract",
    "provider_primitive_kind",
    "provider_widening_product_multiplicand_roles",
    "provider_widening_product_extension_policy",
    "provider_primitive_source_load",
    "provider_primitive_source_extension",
    "provider_primitive_source_dtype",
    "provider_primitive_source_signedness",
    "provider_primitive_source_sew",
    "provider_primitive_source_lmul",
    "provider_primitive_product_dtype",
    "provider_primitive_product_sew",
    "provider_primitive_product_lmul",
    "provider_primitive_accumulator_dtype",
    "provider_primitive_accumulator_sew",
    "provider_primitive_accumulator_lmul",
    "provider_primitive_result_dtype",
    "provider_primitive_result_sew",
    "provider_primitive_result_lmul",
    "provider_primitive_widening_product_relation",
    "provider_primitive_product_reduction_chain_relation",
    "provider_primitive_widening_product_intrinsic",
    "provider_primitive_reduction_intrinsic",
    "provider_primitive_scalar_seed_splat_intrinsic",
    "provider_primitive_accumulator_layout",
    "provider_primitive_result_layout",
    "provider_primitive_reduction_store_vl",
    "provider_remediation_handoff_contract",
    "provider_remediation_diagnosis",
    "provider_remediation_measurement_evidence",
    "provider_remediation_decision",
    "provider_remediation_action",
    "provider_remediation_dispatch_preference",
    "provider_remediation_blocker",
    "target_capability_provider_mirror",
    "target_capability_legality_mirror",
    "provider_maturity",
    "provider_maturity_evidence",
    "provider_maturity_outcome",
    "provider_performance_selection_eligible",
    "provider_dispatch_preference",
    "provider_performance_action",
    "performance_preference_denied",
    "performance_preference_denial_reason",
    "performance_win_claim_allowed",
    "correctness_execution_allowed",
    "provider_contract_update_required",
    "route_support_effect",
)
PACKED_I4_SSH_TARGET_PROFILE = "ssh rvv"
PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH = "performance-preferred"


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


def baseline_identity_for(
    expectation: abi.OpExpectation, *, uses_packed_i4_resource: bool
) -> str:
    if uses_packed_i4_resource:
        identity = PACKED_I4_BASELINE_IDENTITIES.get(expectation.kind)
        if not identity:
            raise abi.EvidenceError(
                f"{expectation.kind} has no packed-i4 same-target baseline"
            )
        return identity
    return BASELINE_IDENTITIES[expectation.kind]


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
    uses_packed_i4_resource: bool,
) -> dict[str, str]:
    baseline_identity = baseline_identity_for(
        expectation, uses_packed_i4_resource=uses_packed_i4_resource
    )
    return {
        "header_file": header_file_name,
        "op_kind": expectation.kind,
        "function_name": expectation.function_name,
        "baseline_identity": baseline_identity,
        "baseline_identity_c": c_string_literal(baseline_identity),
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
    uses_packed_i4_resource: bool = False,
) -> str:
    values = common_measurement_prefix(
        header_file_name=header_file_name,
        expectation=expectation,
        config=config,
        compile_flags_summary=compile_flags_summary,
        uses_packed_i4_resource=uses_packed_i4_resource,
    )
    if expectation.is_widening_product_reduce_dequantize_f32:
        if uses_packed_i4_resource:
            return Template(PACKED_I4_DEQUANT_MEASUREMENT_HARNESS_TEMPLATE).substitute(
                values
            )
        return Template(DEQUANT_MEASUREMENT_HARNESS_TEMPLATE).substitute(values)
    if expectation.is_widening_product_reduce_dequant_clamp_f32:
        if uses_packed_i4_resource:
            return Template(PACKED_I4_DEQUANT_CLAMP_MEASUREMENT_HARNESS_TEMPLATE).substitute(
                values
            )
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


PACKED_I4_DEQUANT_MEASUREMENT_HARNESS_TEMPLATE = r'''
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
  baseline_product_reduction_dequant_packed_i4_v1(lhs, rhs, acc, scale,
                                                  baseline_out, n);
  $function_name(lhs, rhs, acc, scale, generated_out, n);

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
            "$op_kind packed-i4 correctness mismatch n=%zu pattern=%d "
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
            "$op_kind packed-i4 coverage missing n=%zu pattern=%d scale=%.9g "
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
              "$op_kind packed-i4 touched tail sentinel n=%zu pattern=%d "
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
              "$op_kind packed-i4 source buffer mutated n=%zu pattern=%d "
              "scale=%.9g index=%zu lhs=%d before=%d rhs=%d before=%d\n",
              n, pattern, scale, index, lhs[index], lhs_before[index],
              rhs[index], rhs_before[index]);
      return 15;
    }
  }
  for (size_t index = 0; index < 4; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "$op_kind packed-i4 accumulator buffer mutated n=%zu "
              "pattern=%d scale=%.9g index=%zu got=%d before=%d\n",
              n, pattern, scale, index, acc[index], acc_before[index]);
      return 16;
    }
  }
  printf("CORRECTNESS op=$op_kind n=%zu pattern=%d scale=%.9g ok "
         "baseline=$baseline_identity generated=$generated_identity "
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
    fprintf(stderr, "allocation failed for $op_kind packed-i4 n=%zu\n", n);
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
         "scale=%.9g status=passed packed_i4_reference_oracle\n",
         n, pattern, scale);

  for (int warmup = 0; warmup < MEASURE_WARMUPS; ++warmup) {
    baseline_product_reduction_dequant_packed_i4_v1(lhs, rhs, acc, scale,
                                                    baseline_out, n);
    $function_name(lhs, rhs, acc, scale, generated_out, n);
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
           "generated_per_iter_ns=%.3f speedup=%.6f "
           "packed_i4_reference_oracle\n",
           n, pattern, scale, repeat, baseline_ns, generated_ns,
           baseline_per_iter, generated_per_iter, speedup);
  }
  printf("SUMMARY op=$op_kind n=%zu pattern=%d scale=%.9g "
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
         "iterations=%d correctness_before_timing=true "
         "packed_i4_reference_oracle=true runtime_n_unit=packed_bytes\n",
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
         "scales=$scales_summary baseline=$baseline_identity "
         "generated=$generated_identity timing_method=$timing_method "
         "warmups=%d repeats=%d iterations=%d packed_i4_reference_oracle "
         "sink=%.9g\n",
         MEASURE_WARMUPS, MEASURE_REPEATS, MEASURE_ITERATIONS,
         tcrv_measurement_sink);
  return 0;
}
'''.lstrip()


PACKED_I4_DEQUANT_CLAMP_MEASUREMENT_HARNESS_TEMPLATE = r'''
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
baseline_product_reduction_dequant_clamp_packed_i4_v1(
    const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale,
    float lower_bound, float upper_bound, float *out, size_t n) {
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
    lhs[index] = pack_signed_i4_pair(lhs_low_value(index, pattern),
                                     lhs_high_value(index, pattern));
    rhs[index] = pack_signed_i4_pair(rhs_low_value(index, pattern),
                                     rhs_high_value(index, pattern));
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
  baseline_product_reduction_dequant_clamp_packed_i4_v1(
      lhs, rhs, acc, scale, lower_bound, upper_bound, baseline_out, n);
  $function_name(lhs, rhs, acc, scale, lower_bound, upper_bound, generated_out,
                 n);

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
            "$op_kind packed-i4 correctness mismatch n=%zu pattern=%d "
            "scale=%.9g bounds=[%.9g,%.9g] baseline=%.9g generated=%.9g "
            "expected=%.9g scaled=%.9g baseline_delta=%.9g "
            "generated_delta=%.9g tolerance=%.9g expected_acc=%d\n",
            n, pattern, scale, lower_bound, upper_bound, baseline_out[0],
            generated_out[0], expected, scaled, baseline_delta,
            generated_delta, tolerance, expected_acc);
    return 12;
  }
  if (n > 1 &&
      (signed_positive_products == 0 || signed_negative_products == 0 ||
       low_nibble_products == 0 || high_nibble_products == 0)) {
    fprintf(stderr,
            "$op_kind packed-i4 coverage missing n=%zu pattern=%d scale=%.9g "
            "bounds=[%.9g,%.9g] positive=%zu negative=%zu low=%zu high=%zu\n",
            n, pattern, scale, lower_bound, upper_bound,
            signed_positive_products, signed_negative_products,
            low_nibble_products, high_nibble_products);
    return 13;
  }
  for (size_t index = 1; index < 16; ++index) {
    if (generated_out[index] != old_generated_out[index] ||
        baseline_out[index] != old_baseline_out[index]) {
      fprintf(stderr,
              "$op_kind packed-i4 touched tail sentinel n=%zu pattern=%d "
              "scale=%.9g bounds=[%.9g,%.9g] index=%zu generated=%.9g "
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
              "$op_kind packed-i4 source buffer mutated n=%zu pattern=%d "
              "scale=%.9g bounds=[%.9g,%.9g] index=%zu lhs=%d before=%d "
              "rhs=%d before=%d\n",
              n, pattern, scale, lower_bound, upper_bound, index, lhs[index],
              lhs_before[index], rhs[index], rhs_before[index]);
      return 15;
    }
  }
  for (size_t index = 0; index < 4; ++index) {
    if (acc[index] != acc_before[index]) {
      fprintf(stderr,
              "$op_kind packed-i4 accumulator buffer mutated n=%zu pattern=%d "
              "scale=%.9g bounds=[%.9g,%.9g] index=%zu got=%d before=%d\n",
              n, pattern, scale, lower_bound, upper_bound, index, acc[index],
              acc_before[index]);
      return 16;
    }
  }
  printf("CORRECTNESS op=$op_kind n=%zu pattern=%d scale=%.9g "
         "bounds=[%.9g,%.9g] ok baseline=$baseline_identity "
         "generated=$generated_identity expected_acc=%d packed_i4_reference_oracle "
         "source_preserved accumulator_preserved tail_preserved\n",
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
    fprintf(stderr, "allocation failed for $op_kind packed-i4 n=%zu\n", n);
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
         "scale=%.9g bounds=%.9g:%.9g status=passed "
         "packed_i4_reference_oracle\n",
         n, pattern, scale, bounds.lower_bound, bounds.upper_bound);

  for (int warmup = 0; warmup < MEASURE_WARMUPS; ++warmup) {
    baseline_product_reduction_dequant_clamp_packed_i4_v1(
        lhs, rhs, acc, scale, bounds.lower_bound, bounds.upper_bound,
        baseline_out, n);
    $function_name(lhs, rhs, acc, scale, bounds.lower_bound,
                   bounds.upper_bound, generated_out, n);
    tcrv_measurement_sink += (double)baseline_out[0] + (double)generated_out[0];
  }

  double best_baseline_per_iter = -1.0;
  double best_generated_per_iter = -1.0;
  for (int repeat = 0; repeat < MEASURE_REPEATS; ++repeat) {
    unsigned long long start = now_ns();
    for (int iter = 0; iter < MEASURE_ITERATIONS; ++iter) {
      baseline_product_reduction_dequant_clamp_packed_i4_v1(
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
           "bounds=%.9g:%.9g repeat=%d baseline_ns=%llu generated_ns=%llu "
           "baseline_per_iter_ns=%.3f generated_per_iter_ns=%.3f "
           "speedup=%.6f packed_i4_reference_oracle\n",
           n, pattern, scale, bounds.lower_bound, bounds.upper_bound, repeat,
           baseline_ns, generated_ns, baseline_per_iter, generated_per_iter,
           speedup);
  }
  printf("SUMMARY op=$op_kind n=%zu pattern=%d scale=%.9g bounds=%.9g:%.9g "
         "baseline_best_per_iter_ns=%.3f generated_best_per_iter_ns=%.3f "
         "best_speedup=%.6f warmups=%d repeats=%d iterations=%d "
         "packed_i4_reference_oracle\n",
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
         "iterations=%d correctness_before_timing=true "
         "packed_i4_reference_oracle=true runtime_n_unit=packed_bytes\n",
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
         "packed_i4_reference_oracle sink=%.9g\n",
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


def parse_float_record_field(
    record: dict[str, str], field: str, context: str
) -> float:
    value = record.get(field)
    if value is None:
        raise abi.EvidenceError(f"{context} missing {field}")
    try:
        return float(value)
    except ValueError as exc:
        raise abi.EvidenceError(
            f"{context} has non-numeric {field}: {value}"
        ) from exc


def not_measured_result_classification(reason: str) -> dict[str, Any]:
    return {
        "classification": RESULT_CLASSIFICATION_NOT_MEASURED,
        "outcome_family": RESULT_CLASSIFICATION_NOT_MEASURED,
        "reason": reason,
        "timing_method": TIMING_METHOD,
        "correctness_before_timing": True,
    }


def classify_parsed_timing(parsed_timing: dict[str, Any]) -> dict[str, Any]:
    summary_records = parsed_timing.get("summary_records", [])
    if not summary_records:
        raise abi.EvidenceError(
            "same-target measurement output did not contain SUMMARY records"
        )
    case_summaries: list[dict[str, Any]] = []
    best_speedups: list[float] = []
    for index, record in enumerate(summary_records):
        context = f"same-target SUMMARY record {index}"
        best_speedup = parse_float_record_field(record, "best_speedup", context)
        baseline_best = parse_float_record_field(
            record, "baseline_best_per_iter_ns", context
        )
        generated_best = parse_float_record_field(
            record, "generated_best_per_iter_ns", context
        )
        best_speedups.append(best_speedup)
        case_summaries.append(
            {
                "n": record.get("n", ""),
                "pattern": record.get("pattern", ""),
                "scale": record.get("scale", ""),
                "baseline_best_per_iter_ns": baseline_best,
                "generated_best_per_iter_ns": generated_best,
                "best_speedup": best_speedup,
            }
        )

    if all(speedup > 1.0 for speedup in best_speedups):
        classification = RESULT_CLASSIFICATION_WIN
        outcome_family = RESULT_CLASSIFICATION_WIN
    elif all(speedup < 1.0 for speedup in best_speedups):
        classification = RESULT_CLASSIFICATION_REGRESSION
        outcome_family = RESULT_CLASSIFICATION_NO_WIN
    else:
        classification = RESULT_CLASSIFICATION_NO_WIN
        outcome_family = RESULT_CLASSIFICATION_NO_WIN

    speedup_min = min(best_speedups)
    speedup_max = max(best_speedups)
    return {
        "classification": classification,
        "outcome_family": outcome_family,
        "classification_rule": (
            "win iff every parsed SUMMARY best_speedup is > 1.0; "
            "regression iff every parsed SUMMARY best_speedup is < 1.0; "
            "otherwise no-win"
        ),
        "basis": (
            "SUMMARY best_speedup is scalar-baseline best per-iteration time "
            "divided by generated-artifact best per-iteration time"
        ),
        "best_speedup_min": speedup_min,
        "best_speedup_max": speedup_max,
        "best_speedup_range": f"{speedup_min:.6f}..{speedup_max:.6f}",
        "summary_record_count": parsed_timing.get(
            "summary_record_count", len(summary_records)
        ),
        "measurement_record_count": parsed_timing.get("measurement_record_count", 0),
        "correctness_record_count": parsed_timing.get("correctness_record_count", 0),
        "case_summaries": case_summaries,
        "timing_method": TIMING_METHOD,
        "correctness_before_timing": True,
    }


def provider_contract_allows_performance_claim(
    fields: dict[str, str], classification: str
) -> bool:
    return (
        classification == RESULT_CLASSIFICATION_WIN
        and fields["performance_action"]
        != abi.WIDENING_PRODUCT_REDUCE_DEQUANTIZE_F32_PACKED_I4_PERFORMANCE_ACTION
        and fields["performance_selection_eligible"] == "true"
        and fields["dispatch_preference"] == PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH
    )


def performance_preference_denial_reason(
    fields: dict[str, str], classification: str
) -> str:
    if classification == RESULT_CLASSIFICATION_NOT_MEASURED:
        return "same-target-measurement-not-run"
    if classification in (
        RESULT_CLASSIFICATION_NO_WIN,
        RESULT_CLASSIFICATION_REGRESSION,
    ):
        return "same-target-measurement-no-win-or-regression"
    if fields["performance_selection_eligible"] != "true":
        return "provider-contract-performance-selection-ineligible"
    if fields["dispatch_preference"] != PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH:
        return "provider-contract-not-performance-preferred"
    if (
        fields["performance_action"]
        == abi.WIDENING_PRODUCT_REDUCE_DEQUANTIZE_F32_PACKED_I4_PERFORMANCE_ACTION
    ):
        return "provider-contract-requires-no-win-repair"
    return ""


def maturity_contract_alignment(
    *,
    fields: dict[str, str],
    classification: str,
    outcome_family: str,
    performance_win_claim_allowed: bool,
) -> str:
    provider_outcome = fields["performance_maturity_outcome"]
    if classification == RESULT_CLASSIFICATION_NOT_MEASURED:
        return "not-measured"
    if classification == provider_outcome:
        return "matches-provider-maturity-outcome"
    if (
        outcome_family == RESULT_CLASSIFICATION_NO_WIN
        and provider_outcome
        in (RESULT_CLASSIFICATION_NO_WIN, RESULT_CLASSIFICATION_REGRESSION)
    ):
        return "same-no-win-family-denies-performance-preference"
    if classification == RESULT_CLASSIFICATION_WIN:
        if performance_win_claim_allowed:
            return "win-claim-allowed-by-provider-maturity-contract"
        return (
            "measurement-win-conflicts-with-provider-maturity-contract-requires-"
            "provider-update"
        )
    return "measurement-outcome-requires-provider-maturity-review"


def runtime_count_set(counts: list[int]) -> str:
    return ",".join(str(count) for count in counts)


def source_backed_pressure_profile_record_context(
    *,
    generation_result: dict[str, Any],
    expectation: abi.OpExpectation,
    object_path: Path,
    header_path: Path,
    config: MeasurementConfig,
    result_classification: dict[str, Any],
) -> dict[str, Any]:
    classification = str(result_classification.get("classification", ""))
    measured = classification != RESULT_CLASSIFICATION_NOT_MEASURED
    return {
        "source_record_contract": SOURCE_BACKED_MEASUREMENT_RECORD_CONTRACT,
        "source_selected_variant": expectation.selected_variant,
        "source_selected_input": generation_result["selected_input"]["path"],
        "source_generated_function": expectation.function_name,
        "generated_artifact_identity_contract": (
            GENERATED_ARTIFACT_IDENTITY_CONTRACT
        ),
        "generated_artifact_object_path": str(object_path),
        "generated_artifact_object_sha256": abi.sha256_file(object_path),
        "generated_artifact_header_path": str(header_path),
        "generated_artifact_header_sha256": abi.sha256_file(header_path),
        "measurement_target": PACKED_I4_SSH_TARGET_PROFILE if measured else "",
        "measurement_target_provenance": MEASUREMENT_TARGET_PROVENANCE,
        "measurement_runtime_count_set": runtime_count_set(config.counts),
        "measurement_runtime_count_provenance": (
            MEASUREMENT_RUNTIME_COUNT_PROVENANCE
        ),
        "pressure_profile_label": PRODUCTION_PRESSURE_PROFILE_LABEL,
        "pressure_profile_label_provenance": (
            PRODUCTION_PRESSURE_PROFILE_LABEL_PROVENANCE
        ),
    }


def packed_i4_resource_int(fields: dict[str, str], name: str) -> int:
    try:
        return int(fields[name])
    except KeyError as exc:
        raise abi.EvidenceError(
            f"packed-i4 provider feedback tie-back missing {name}"
        ) from exc
    except ValueError as exc:
        raise abi.EvidenceError(
            f"packed-i4 provider feedback tie-back requires integer {name}: "
            f"{fields.get(name)!r}"
        ) from exc


def packed_i4_maturity_contract_evidence_input(
    *,
    fields: dict[str, str],
    result_classification: dict[str, Any],
    measurement_evidence_id: str,
    source_record_context: dict[str, Any],
) -> dict[str, Any]:
    classification = str(result_classification.get("classification", ""))
    outcome_family = str(result_classification.get("outcome_family", ""))
    performance_win_claim_allowed = provider_contract_allows_performance_claim(
        fields, classification
    )
    alignment = maturity_contract_alignment(
        fields=fields,
        classification=classification,
        outcome_family=outcome_family,
        performance_win_claim_allowed=performance_win_claim_allowed,
    )
    denial_reason = performance_preference_denial_reason(fields, classification)
    return {
        "contract": PACKED_I4_MATURITY_CONTRACT_EVIDENCE_INPUT,
        "authority": PACKED_I4_MATURITY_CONTRACT_AUTHORITY,
        "measurement_evidence_id": measurement_evidence_id,
        "measurement_classification": classification,
        "measurement_outcome_family": outcome_family,
        "measurement_best_speedup_range": result_classification.get(
            "best_speedup_range", ""
        ),
        "measurement_summary_record_count": result_classification.get(
            "summary_record_count", 0
        ),
        "measurement_record_count": result_classification.get(
            "measurement_record_count", 0
        ),
        "correctness_record_count": result_classification.get(
            "correctness_record_count", 0
        ),
        "same_target_measurement": classification
        != RESULT_CLASSIFICATION_NOT_MEASURED,
        "ssh_evidence": classification != RESULT_CLASSIFICATION_NOT_MEASURED,
        "target_profile": (
            PACKED_I4_SSH_TARGET_PROFILE
            if classification != RESULT_CLASSIFICATION_NOT_MEASURED
            else ""
        ),
        "source_record_contract": source_record_context["source_record_contract"],
        "source_selected_variant": source_record_context[
            "source_selected_variant"
        ],
        "source_selected_input": source_record_context["source_selected_input"],
        "source_generated_function": source_record_context[
            "source_generated_function"
        ],
        "generated_artifact_identity_contract": source_record_context[
            "generated_artifact_identity_contract"
        ],
        "generated_artifact_object_path": source_record_context[
            "generated_artifact_object_path"
        ],
        "generated_artifact_object_sha256": source_record_context[
            "generated_artifact_object_sha256"
        ],
        "generated_artifact_header_path": source_record_context[
            "generated_artifact_header_path"
        ],
        "generated_artifact_header_sha256": source_record_context[
            "generated_artifact_header_sha256"
        ],
        "measurement_target": source_record_context["measurement_target"],
        "measurement_target_provenance": source_record_context[
            "measurement_target_provenance"
        ],
        "measurement_runtime_count_set": source_record_context[
            "measurement_runtime_count_set"
        ],
        "measurement_runtime_count_provenance": source_record_context[
            "measurement_runtime_count_provenance"
        ],
        "pressure_profile_label": source_record_context[
            "pressure_profile_label"
        ],
        "pressure_profile_label_provenance": source_record_context[
            "pressure_profile_label_provenance"
        ],
        "provider_maturity": fields["performance_maturity"],
        "provider_maturity_evidence": fields["performance_maturity_evidence"],
        "provider_maturity_outcome": fields["performance_maturity_outcome"],
        "provider_resource_selected_candidate": fields["selected_candidate"],
        "provider_resource_planning_contract": fields["planning_contract"],
        "provider_resource_operand_form": fields["operand_form"],
        "provider_resource_source_signedness": fields["source_signedness"],
        "provider_resource_storage_element_width": packed_i4_resource_int(
            fields, "storage_element_width"
        ),
        "provider_resource_effective_element_width": packed_i4_resource_int(
            fields, "effective_element_width"
        ),
        "provider_resource_packing_layout": fields["packing_layout"],
        "provider_resource_unpack_intent": fields["unpack_intent"],
        "provider_resource_vsetvl_region_count": packed_i4_resource_int(
            fields, "vsetvl_region_count"
        ),
        "provider_runtime_avl_source": fields["runtime_avl_source"],
        "provider_resource_route_family_plan": fields["route_family_plan"],
        "provider_supported_mirror": fields["provider_supported_mirror"],
        "provider_runtime_abi_order": fields["runtime_abi_order"],
        "provider_primitive_chain_contract": fields["primitive_chain_contract"],
        "provider_primitive_chain_kind": fields["primitive_chain_kind"],
        "provider_primitive_contract": fields["primitive_contract"],
        "provider_primitive_kind": fields["primitive_kind"],
        "provider_widening_product_multiplicand_roles": fields[
            "widening_product_multiplicand_roles"
        ],
        "provider_widening_product_extension_policy": fields[
            "widening_product_extension_policy"
        ],
        "provider_primitive_source_load": fields["primitive_source_load"],
        "provider_primitive_source_extension": fields[
            "primitive_source_extension"
        ],
        "provider_primitive_source_dtype": fields["source_dtype"],
        "provider_primitive_source_signedness": fields["source_signedness"],
        "provider_primitive_source_sew": packed_i4_resource_int(
            fields, "source_sew"
        ),
        "provider_primitive_source_lmul": fields["source_lmul"],
        "provider_primitive_product_dtype": fields["product_dtype"],
        "provider_primitive_product_sew": packed_i4_resource_int(
            fields, "product_sew"
        ),
        "provider_primitive_product_lmul": fields["product_lmul"],
        "provider_primitive_accumulator_dtype": fields["accumulator_dtype"],
        "provider_primitive_accumulator_sew": packed_i4_resource_int(
            fields, "accumulator_sew"
        ),
        "provider_primitive_accumulator_lmul": fields["accumulator_lmul"],
        "provider_primitive_result_dtype": fields["result_dtype"],
        "provider_primitive_result_sew": packed_i4_resource_int(
            fields, "result_sew"
        ),
        "provider_primitive_result_lmul": fields["result_lmul"],
        "provider_primitive_widening_product_relation": fields[
            "primitive_widening_product_relation"
        ],
        "provider_primitive_product_reduction_chain_relation": fields[
            "primitive_product_reduction_chain_relation"
        ],
        "provider_primitive_widening_product_intrinsic": fields[
            "primitive_widening_product_intrinsic"
        ],
        "provider_primitive_reduction_intrinsic": fields[
            "primitive_reduction_intrinsic"
        ],
        "provider_primitive_scalar_seed_splat_intrinsic": fields[
            "primitive_scalar_seed_splat_intrinsic"
        ],
        "provider_primitive_accumulator_layout": fields[
            "primitive_accumulator_layout"
        ],
        "provider_primitive_result_layout": fields["primitive_result_layout"],
        "provider_primitive_reduction_store_vl": fields[
            "primitive_reduction_store_vl"
        ],
        "provider_remediation_handoff_contract": fields[
            "remediation_handoff_contract"
        ],
        "provider_remediation_diagnosis": fields["remediation_diagnosis"],
        "provider_remediation_measurement_evidence": fields[
            "remediation_measurement_evidence"
        ],
        "provider_remediation_decision": fields["remediation_decision"],
        "provider_remediation_action": fields["remediation_action"],
        "provider_remediation_dispatch_preference": fields[
            "remediation_dispatch_preference"
        ],
        "provider_remediation_blocker": fields["remediation_blocker"],
        "provider_remediation_plan_contract": fields[
            "remediation_plan_contract"
        ],
        "provider_remediation_plan": fields["remediation_plan"],
        "provider_remediation_statement_strategy": fields[
            "remediation_statement_strategy"
        ],
        "provider_remediation_vector_budget": fields[
            "remediation_vector_budget"
        ],
        "provider_remediation_schedule_contract": fields[
            "remediation_schedule_contract"
        ],
        "provider_remediation_unpack_plan": fields["remediation_unpack_plan"],
        "provider_remediation_product_plan": fields["remediation_product_plan"],
        "provider_remediation_reduction_plan": fields[
            "remediation_reduction_plan"
        ],
        "provider_remediation_vl_plan": fields["remediation_vl_plan"],
        "provider_schedule_decision_contract": fields[
            "schedule_decision_contract"
        ],
        "provider_schedule_decision": fields["schedule_decision"],
        "provider_schedule_decision_reason": fields["schedule_decision_reason"],
        "provider_resource_cost_contract": fields["resource_cost_contract"],
        "provider_resource_cost_model": fields["resource_cost_model"],
        "provider_resource_cost_loop_body_steps": packed_i4_resource_int(
            fields, "resource_cost_loop_body_steps"
        ),
        "provider_resource_cost_blocker": fields["resource_cost_blocker"],
        "provider_performance_admission_decision": fields[
            "performance_admission_decision"
        ],
        "provider_realization_admission_contract": fields[
            "realization_admission_contract"
        ],
        "provider_realization_admission_decision": fields[
            "realization_admission_decision"
        ],
        "provider_realization_admission_evidence": fields[
            "realization_admission_evidence"
        ],
        "provider_realization_admission_dispatch_policy": fields[
            "realization_admission_dispatch_policy"
        ],
        "provider_realization_admission_schedule_decision_contract": fields[
            "realization_admission_schedule_decision_contract"
        ],
        "provider_realization_admission_schedule_decision": fields[
            "realization_admission_schedule_decision"
        ],
        "provider_realization_admission_schedule_decision_reason": fields[
            "realization_admission_schedule_decision_reason"
        ],
        "target_capability_provider_mirror": fields[
            "target_capability_provider_mirror"
        ],
        "target_capability_legality_mirror": fields[
            "target_capability_legality_mirror"
        ],
        "provider_performance_selection_eligible": fields[
            "performance_selection_eligible"
        ],
        "provider_dispatch_preference": fields["dispatch_preference"],
        "provider_performance_action": fields["performance_action"],
        "contract_alignment": alignment,
        "performance_win_claim_allowed": performance_win_claim_allowed,
        "performance_preference_denied": not performance_win_claim_allowed,
        "performance_preference_denial_reason": denial_reason,
        "correctness_execution_allowed": True,
        "route_support_effect": (
            "preserve-executable-route-support; measurement evidence only "
            "gates performance preference and claims"
        ),
        "provider_contract_update_required": alignment
        not in ("not-measured", "matches-provider-maturity-outcome"),
    }


def packed_i4_same_target_measurement_record(
    maturity_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        field: maturity_input[field]
        for field in PACKED_I4_SAME_TARGET_MEASUREMENT_RECORD_FIELDS
    }


def require_maturity_input_value(
    maturity_input: dict[str, Any],
    field: str,
    expected: Any,
    context: str,
) -> None:
    actual = maturity_input.get(field)
    if actual != expected:
        raise abi.EvidenceError(
            f"{context} stale maturity-contract evidence field {field}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_packed_i4_maturity_contract_evidence_input(
    *,
    fields: dict[str, str],
    result_classification: dict[str, Any],
    measurement_evidence_id: str,
    source_record_context: dict[str, Any],
    maturity_input: dict[str, Any],
    context: str,
) -> None:
    classification = str(result_classification.get("classification", ""))
    outcome_family = str(result_classification.get("outcome_family", ""))
    performance_win_claim_allowed = provider_contract_allows_performance_claim(
        fields, classification
    )
    expected_alignment = maturity_contract_alignment(
        fields=fields,
        classification=classification,
        outcome_family=outcome_family,
        performance_win_claim_allowed=performance_win_claim_allowed,
    )
    expected_denial_reason = performance_preference_denial_reason(
        fields, classification
    )

    expected_values: dict[str, Any] = {
        "contract": PACKED_I4_MATURITY_CONTRACT_EVIDENCE_INPUT,
        "authority": PACKED_I4_MATURITY_CONTRACT_AUTHORITY,
        "measurement_evidence_id": measurement_evidence_id,
        "measurement_classification": classification,
        "measurement_outcome_family": outcome_family,
        "measurement_best_speedup_range": result_classification.get(
            "best_speedup_range", ""
        ),
        "measurement_summary_record_count": result_classification.get(
            "summary_record_count", 0
        ),
        "measurement_record_count": result_classification.get(
            "measurement_record_count", 0
        ),
        "correctness_record_count": result_classification.get(
            "correctness_record_count", 0
        ),
        "same_target_measurement": classification
        != RESULT_CLASSIFICATION_NOT_MEASURED,
        "ssh_evidence": classification != RESULT_CLASSIFICATION_NOT_MEASURED,
        "target_profile": (
            PACKED_I4_SSH_TARGET_PROFILE
            if classification != RESULT_CLASSIFICATION_NOT_MEASURED
            else ""
        ),
        "source_record_contract": source_record_context["source_record_contract"],
        "source_selected_variant": source_record_context[
            "source_selected_variant"
        ],
        "source_selected_input": source_record_context["source_selected_input"],
        "source_generated_function": source_record_context[
            "source_generated_function"
        ],
        "generated_artifact_identity_contract": source_record_context[
            "generated_artifact_identity_contract"
        ],
        "generated_artifact_object_path": source_record_context[
            "generated_artifact_object_path"
        ],
        "generated_artifact_object_sha256": source_record_context[
            "generated_artifact_object_sha256"
        ],
        "generated_artifact_header_path": source_record_context[
            "generated_artifact_header_path"
        ],
        "generated_artifact_header_sha256": source_record_context[
            "generated_artifact_header_sha256"
        ],
        "measurement_target": source_record_context["measurement_target"],
        "measurement_target_provenance": source_record_context[
            "measurement_target_provenance"
        ],
        "measurement_runtime_count_set": source_record_context[
            "measurement_runtime_count_set"
        ],
        "measurement_runtime_count_provenance": source_record_context[
            "measurement_runtime_count_provenance"
        ],
        "pressure_profile_label": source_record_context[
            "pressure_profile_label"
        ],
        "pressure_profile_label_provenance": source_record_context[
            "pressure_profile_label_provenance"
        ],
        "provider_maturity": fields["performance_maturity"],
        "provider_maturity_evidence": fields["performance_maturity_evidence"],
        "provider_maturity_outcome": fields["performance_maturity_outcome"],
        "provider_resource_selected_candidate": fields["selected_candidate"],
        "provider_resource_planning_contract": fields["planning_contract"],
        "provider_resource_operand_form": fields["operand_form"],
        "provider_resource_source_signedness": fields["source_signedness"],
        "provider_resource_storage_element_width": packed_i4_resource_int(
            fields, "storage_element_width"
        ),
        "provider_resource_effective_element_width": packed_i4_resource_int(
            fields, "effective_element_width"
        ),
        "provider_resource_packing_layout": fields["packing_layout"],
        "provider_resource_unpack_intent": fields["unpack_intent"],
        "provider_resource_vsetvl_region_count": packed_i4_resource_int(
            fields, "vsetvl_region_count"
        ),
        "provider_runtime_avl_source": fields["runtime_avl_source"],
        "provider_resource_route_family_plan": fields["route_family_plan"],
        "provider_supported_mirror": fields["provider_supported_mirror"],
        "provider_runtime_abi_order": fields["runtime_abi_order"],
        "provider_primitive_chain_contract": fields["primitive_chain_contract"],
        "provider_primitive_chain_kind": fields["primitive_chain_kind"],
        "provider_primitive_contract": fields["primitive_contract"],
        "provider_primitive_kind": fields["primitive_kind"],
        "provider_widening_product_multiplicand_roles": fields[
            "widening_product_multiplicand_roles"
        ],
        "provider_widening_product_extension_policy": fields[
            "widening_product_extension_policy"
        ],
        "provider_primitive_source_load": fields["primitive_source_load"],
        "provider_primitive_source_extension": fields[
            "primitive_source_extension"
        ],
        "provider_primitive_source_dtype": fields["source_dtype"],
        "provider_primitive_source_signedness": fields["source_signedness"],
        "provider_primitive_source_sew": packed_i4_resource_int(
            fields, "source_sew"
        ),
        "provider_primitive_source_lmul": fields["source_lmul"],
        "provider_primitive_product_dtype": fields["product_dtype"],
        "provider_primitive_product_sew": packed_i4_resource_int(
            fields, "product_sew"
        ),
        "provider_primitive_product_lmul": fields["product_lmul"],
        "provider_primitive_accumulator_dtype": fields["accumulator_dtype"],
        "provider_primitive_accumulator_sew": packed_i4_resource_int(
            fields, "accumulator_sew"
        ),
        "provider_primitive_accumulator_lmul": fields["accumulator_lmul"],
        "provider_primitive_result_dtype": fields["result_dtype"],
        "provider_primitive_result_sew": packed_i4_resource_int(
            fields, "result_sew"
        ),
        "provider_primitive_result_lmul": fields["result_lmul"],
        "provider_primitive_widening_product_relation": fields[
            "primitive_widening_product_relation"
        ],
        "provider_primitive_product_reduction_chain_relation": fields[
            "primitive_product_reduction_chain_relation"
        ],
        "provider_primitive_widening_product_intrinsic": fields[
            "primitive_widening_product_intrinsic"
        ],
        "provider_primitive_reduction_intrinsic": fields[
            "primitive_reduction_intrinsic"
        ],
        "provider_primitive_scalar_seed_splat_intrinsic": fields[
            "primitive_scalar_seed_splat_intrinsic"
        ],
        "provider_primitive_accumulator_layout": fields[
            "primitive_accumulator_layout"
        ],
        "provider_primitive_result_layout": fields["primitive_result_layout"],
        "provider_primitive_reduction_store_vl": fields[
            "primitive_reduction_store_vl"
        ],
        "provider_remediation_handoff_contract": fields[
            "remediation_handoff_contract"
        ],
        "provider_remediation_diagnosis": fields["remediation_diagnosis"],
        "provider_remediation_measurement_evidence": fields[
            "remediation_measurement_evidence"
        ],
        "provider_remediation_decision": fields["remediation_decision"],
        "provider_remediation_action": fields["remediation_action"],
        "provider_remediation_dispatch_preference": fields[
            "remediation_dispatch_preference"
        ],
        "provider_remediation_blocker": fields["remediation_blocker"],
        "provider_remediation_plan_contract": fields[
            "remediation_plan_contract"
        ],
        "provider_remediation_plan": fields["remediation_plan"],
        "provider_remediation_statement_strategy": fields[
            "remediation_statement_strategy"
        ],
        "provider_remediation_vector_budget": fields[
            "remediation_vector_budget"
        ],
        "provider_remediation_schedule_contract": fields[
            "remediation_schedule_contract"
        ],
        "provider_remediation_unpack_plan": fields["remediation_unpack_plan"],
        "provider_remediation_product_plan": fields["remediation_product_plan"],
        "provider_remediation_reduction_plan": fields[
            "remediation_reduction_plan"
        ],
        "provider_remediation_vl_plan": fields["remediation_vl_plan"],
        "provider_schedule_decision_contract": fields[
            "schedule_decision_contract"
        ],
        "provider_schedule_decision": fields["schedule_decision"],
        "provider_schedule_decision_reason": fields["schedule_decision_reason"],
        "provider_resource_cost_contract": fields["resource_cost_contract"],
        "provider_resource_cost_model": fields["resource_cost_model"],
        "provider_resource_cost_loop_body_steps": packed_i4_resource_int(
            fields, "resource_cost_loop_body_steps"
        ),
        "provider_resource_cost_blocker": fields["resource_cost_blocker"],
        "provider_performance_admission_decision": fields[
            "performance_admission_decision"
        ],
        "provider_realization_admission_contract": fields[
            "realization_admission_contract"
        ],
        "provider_realization_admission_decision": fields[
            "realization_admission_decision"
        ],
        "provider_realization_admission_evidence": fields[
            "realization_admission_evidence"
        ],
        "provider_realization_admission_dispatch_policy": fields[
            "realization_admission_dispatch_policy"
        ],
        "provider_realization_admission_schedule_decision_contract": fields[
            "realization_admission_schedule_decision_contract"
        ],
        "provider_realization_admission_schedule_decision": fields[
            "realization_admission_schedule_decision"
        ],
        "provider_realization_admission_schedule_decision_reason": fields[
            "realization_admission_schedule_decision_reason"
        ],
        "target_capability_provider_mirror": fields[
            "target_capability_provider_mirror"
        ],
        "target_capability_legality_mirror": fields[
            "target_capability_legality_mirror"
        ],
        "provider_performance_selection_eligible": fields[
            "performance_selection_eligible"
        ],
        "provider_dispatch_preference": fields["dispatch_preference"],
        "provider_performance_action": fields["performance_action"],
        "contract_alignment": expected_alignment,
        "performance_win_claim_allowed": performance_win_claim_allowed,
        "performance_preference_denied": not performance_win_claim_allowed,
        "performance_preference_denial_reason": expected_denial_reason,
        "correctness_execution_allowed": True,
        "provider_contract_update_required": expected_alignment
        not in ("not-measured", "matches-provider-maturity-outcome"),
    }
    for field, expected in expected_values.items():
        require_maturity_input_value(maturity_input, field, expected, context)

    if classification in (
        RESULT_CLASSIFICATION_NOT_MEASURED,
        RESULT_CLASSIFICATION_NO_WIN,
        RESULT_CLASSIFICATION_REGRESSION,
    ):
        if maturity_input.get("performance_win_claim_allowed") is not False:
            raise abi.EvidenceError(
                f"{context} must fail closed: {classification} evidence "
                "cannot allow a performance-win claim"
            )
        if maturity_input.get("performance_preference_denied") is not True:
            raise abi.EvidenceError(
                f"{context} must deny performance preference for "
                f"{classification} evidence"
            )
        if fields["performance_selection_eligible"] == "true":
            raise abi.EvidenceError(
                f"{context} must fail closed: {classification} evidence "
                "cannot be paired with provider performance selection "
                "eligibility"
            )
        if (
            fields["dispatch_preference"]
            == PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH
        ):
            raise abi.EvidenceError(
                f"{context} must fail closed: {classification} evidence "
                "cannot be paired with provider performance-preferred dispatch"
            )


def packed_i4_provider_feedback_tie_back(
    *,
    generation_result: dict[str, Any],
    expectation: abi.OpExpectation,
    uses_packed_i4_resource: bool,
    result_classification: dict[str, Any],
    measurement_evidence_id: str,
    source_record_context: dict[str, Any],
) -> dict[str, Any]:
    if not uses_packed_i4_resource:
        return {
            "packed_i4_resource_metadata_selected": False,
            "baseline_identity": baseline_identity_for(
                expectation, uses_packed_i4_resource=False
            ),
            "status": "not-applicable",
        }

    boundary = generation_result.get("widening_product_reduction_boundary", {})
    route_metadata = boundary.get("route_metadata", {})
    provider_low_precision = (
        boundary.get("provider_route_facts", {}).get("low_precision_resource", {})
    )
    target_artifact_metadata: dict[str, str] = {}
    for record in (
        generation_result.get("bundle_checks", {})
        .get("index", {})
        .get("parsed", {})
        .get("records", [])
    ):
        for entry in record.get("artifact_metadata", []):
            key = str(entry.get("key", ""))
            if not key.startswith("tcrv_rvv.low_precision_resource."):
                continue
            value = str(entry.get("value", ""))
            previous = target_artifact_metadata.get(key)
            if previous is not None and previous != value:
                raise abi.EvidenceError(
                    "packed-i4 provider feedback tie-back rejects target "
                    f"artifact metadata disagreement for {key}: "
                    f"{previous!r} vs {value!r}"
                )
            target_artifact_metadata[key] = value

    def resource_field(name: str) -> str:
        route_key = f"tcrv_rvv.low_precision_resource.{name}"
        value = target_artifact_metadata.get(route_key)
        if value is None:
            value = route_metadata.get(route_key)
        if value is None:
            value = provider_low_precision.get(name)
        if value is None:
            raise abi.EvidenceError(
                f"packed-i4 provider feedback tie-back missing {route_key}"
            )
        return str(value)

    expected_resource_metadata = abi.expected_low_precision_resource_metadata(
        expectation, packed_i4=True
    )
    expected_field_names = (
        "selected_candidate",
        "selection_reason",
        "planning_contract",
        "route_family_plan",
        "provider_supported_mirror",
        "runtime_avl_source",
        "runtime_abi_order",
        "source_dtype",
        "source_signedness",
        "source_sew",
        "source_lmul",
        "product_dtype",
        "product_sew",
        "product_lmul",
        "accumulator_dtype",
        "accumulator_sew",
        "accumulator_lmul",
        "result_dtype",
        "result_sew",
        "result_lmul",
        "storage_element_width",
        "effective_element_width",
        "vsetvl_region_count",
        "primitive_contract",
        "primitive_kind",
        "primitive_chain_contract",
        "primitive_chain_kind",
        "widening_product_multiplicand_roles",
        "widening_product_extension_policy",
        "primitive_source_load",
        "primitive_source_extension",
        "primitive_widening_product_relation",
        "primitive_product_reduction_chain_relation",
        "primitive_widening_product_intrinsic",
        "primitive_reduction_intrinsic",
        "primitive_scalar_seed_splat_intrinsic",
        "primitive_accumulator_layout",
        "primitive_result_layout",
        "primitive_reduction_store_vl",
        "realization_producer",
        "realization_decision",
        "realized_unroll_factor",
        "realized_vsetvl_region_count",
        "realized_peak_live_vector_groups",
        "product_region_index",
        "dequant_region_index",
        "product_phase",
        "dequant_phase",
        "target_capability_provider_mirror",
        "target_capability_legality_mirror",
        "performance_feedback",
        "performance_baseline",
        "performance_best_speedup_range",
        "performance_action",
        "performance_maturity",
        "performance_maturity_evidence",
        "performance_maturity_outcome",
        "performance_selection_eligible",
        "dispatch_preference",
        "remediation_handoff_contract",
        "remediation_diagnosis",
        "remediation_measurement_evidence",
        "remediation_decision",
        "remediation_action",
        "remediation_dispatch_preference",
        "remediation_blocker",
        "remediation_plan_contract",
        "remediation_plan",
        "remediation_statement_strategy",
        "remediation_vector_budget",
        "remediation_schedule_contract",
        "remediation_unpack_plan",
        "remediation_product_plan",
        "remediation_reduction_plan",
        "remediation_vl_plan",
        "schedule_decision_contract",
        "schedule_decision",
        "schedule_decision_reason",
        "resource_cost_contract",
        "resource_cost_model",
        "resource_cost_loop_body_steps",
        "resource_cost_blocker",
        "performance_admission_decision",
        "realization_admission_contract",
        "realization_admission_decision",
        "realization_admission_evidence",
        "realization_admission_dispatch_policy",
        "realization_admission_schedule_decision_contract",
        "realization_admission_schedule_decision",
        "realization_admission_schedule_decision_reason",
        "operand_form",
        "packing_layout",
        "unpack_intent",
    )
    expected_fields = {
        name: expected_resource_metadata[
            f"tcrv_rvv.low_precision_resource.{name}"
        ]
        for name in expected_field_names
    }
    fields = {name: resource_field(name) for name in expected_fields}
    for name, expected in expected_fields.items():
        abi.require_equal(
            fields[name],
            expected,
            f"packed-i4 provider feedback tie-back {name}",
        )

    maturity_input = packed_i4_maturity_contract_evidence_input(
        fields=fields,
        result_classification=result_classification,
        measurement_evidence_id=measurement_evidence_id,
        source_record_context=source_record_context,
    )
    validate_packed_i4_maturity_contract_evidence_input(
        fields=fields,
        result_classification=result_classification,
        measurement_evidence_id=measurement_evidence_id,
        source_record_context=source_record_context,
        maturity_input=maturity_input,
        context="packed-i4 provider feedback tie-back",
    )

    return {
        "packed_i4_resource_metadata_selected": True,
        "authority": (
            "provider-owned low-precision resource facts mirrored by generated "
            "object/header metadata after target artifact validation"
        ),
        "fields": fields,
        "expected_fields": expected_fields,
        "baseline_identity": baseline_identity_for(
            expectation, uses_packed_i4_resource=True
        ),
        "result_alignment": maturity_input["contract_alignment"],
        "maturity_contract_evidence_input": maturity_input,
        "same_target_measurement_record": (
            packed_i4_same_target_measurement_record(maturity_input)
        ),
        "performance_win_claim_allowed": maturity_input[
            "performance_win_claim_allowed"
        ],
        "next_repair_owner_if_no_win": (
            "RVV plugin-local Gearbox/resource/statement planning for the "
            "selected packed-i4 product-reduction candidate"
        ),
    }


def make_generation_args(
    op_kind: str, timeout: int, input_path: Path | None
) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=True,
        direct_pre_realized_route_entry=False,
        input=input_path,
        source_seed=False,
        vector_source_front_door=False,
        pre_realized_selected_body=True,
        rhs_broadcast_selected_body=False,
        lmul_m2_selected_body=False,
        op_kind=[op_kind],
        timeout=timeout,
    )


def selected_pre_realized_expectation(
    op_kind: str, input_path: Path | None = None
) -> abi.OpExpectation:
    if op_kind not in DEFAULT_OP_KINDS:
        raise abi.EvidenceError(
            f"same-target measurement supports only {', '.join(DEFAULT_OP_KINDS)}"
        )
    expectation = abi.PRE_REALIZED_SELECTED_BODY_OP_EXPECTATIONS[op_kind]
    if input_path is not None:
        resolved_input = abi.resolve_repo_relative_path(input_path)
        if not resolved_input.exists():
            raise abi.EvidenceError(f"measurement input not found: {input_path}")
        return replace(expectation, input_path=resolved_input)
    if expectation.input_path.is_absolute():
        return expectation
    return replace(expectation, input_path=abi.REPO_ROOT / expectation.input_path)


def uses_packed_i4_resource_from_bundle(
    bundle_checks: dict[str, Any], expectation: abi.OpExpectation
) -> bool:
    if not (
        expectation.is_widening_product_reduce_dequantize_f32
        or expectation.is_widening_product_reduce_dequant_clamp_f32
    ):
        return False
    metadata = abi.widening_product_reduction_metadata_from_bundle(
        bundle_checks, expectation
    )
    return abi.product_dequant_uses_packed_i4_resource_metadata(
        metadata, expectation
    )


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
    generation_args = make_generation_args(
        expectation.kind, args.timeout, args.input
    )
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
    uses_packed_i4_resource: bool,
    result_classification: dict[str, Any],
    provider_feedback_tie_back: dict[str, Any],
) -> dict[str, Any]:
    bundle_checks = generation_result["bundle_checks"]
    bundle_dir = Path(generation_result["artifact_dir"]) / "generated_bundle"
    object_path = bundle_dir / bundle_checks["object_file"]
    header_path = bundle_dir / bundle_checks["header_file"]
    baseline_identity = baseline_identity_for(
        expectation, uses_packed_i4_resource=uses_packed_i4_resource
    )
    summary: dict[str, Any] = {
        "status": "success" if remote else "dry_run_success",
        "op_kind": expectation.kind,
        "baseline_identity": baseline_identity,
        "baseline_role": "same-target scalar C comparator and correctness oracle",
        "packed_i4_resource_metadata_selected": uses_packed_i4_resource,
        "result_classification": result_classification,
        "provider_feedback_tie_back": provider_feedback_tie_back,
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
            "baseline_identity": baseline_identity,
            "packed_i4_reference_oracle": uses_packed_i4_resource,
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
    if uses_packed_i4_resource:
        fields = provider_feedback_tie_back.get("fields", {})
        schedule_decision_evidence = {
            "schedule_decision": fields.get("schedule_decision", ""),
            "provider_schedule_decision_contract": fields.get(
                "schedule_decision_contract", ""
            ),
            "provider_schedule_decision": fields.get("schedule_decision", ""),
            "provider_schedule_decision_reason": fields.get(
                "schedule_decision_reason", ""
            ),
            "provider_resource_cost_contract": fields.get(
                "resource_cost_contract", ""
            ),
            "provider_resource_cost_model": fields.get(
                "resource_cost_model", ""
            ),
            "provider_resource_cost_loop_body_steps": packed_i4_resource_int(
                fields, "resource_cost_loop_body_steps"
            ),
            "provider_resource_cost_blocker": fields.get(
                "resource_cost_blocker", ""
            ),
            "provider_performance_admission_decision": fields.get(
                "performance_admission_decision", ""
            ),
            "provider_realization_admission_schedule_decision_contract": fields.get(
                "realization_admission_schedule_decision_contract", ""
            ),
            "provider_realization_admission_schedule_decision": fields.get(
                "realization_admission_schedule_decision", ""
            ),
            "provider_realization_admission_schedule_decision_reason": fields.get(
                "realization_admission_schedule_decision_reason", ""
            ),
        }
        summary["measurement_harness"].update(schedule_decision_evidence)
        summary["same_target_schedule_decision_evidence"] = (
            schedule_decision_evidence
        )
    maturity_input = provider_feedback_tie_back.get(
        "maturity_contract_evidence_input"
    )
    if maturity_input:
        summary["performance_maturity_contract_evidence_input"] = maturity_input
    same_target_record = provider_feedback_tie_back.get(
        "same_target_measurement_record"
    )
    if same_target_record:
        summary["same_target_measurement_record"] = same_target_record
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
            "result_classification": result_classification,
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
        uses_packed_i4_resource = uses_packed_i4_resource_from_bundle(
            bundle_checks, expectation
        )
        baseline_identity = baseline_identity_for(
            expectation, uses_packed_i4_resource=uses_packed_i4_resource
        )
        evidence["baseline_identity"] = baseline_identity
        evidence["packed_i4_resource_metadata_selected"] = uses_packed_i4_resource
        if uses_packed_i4_resource:
            evidence["packed_i4_reference_oracle"] = {
                "source": (
                    "provider-owned low-precision resource metadata selected "
                    "signed packed-i4 nibbles"
                ),
                "baseline_identity": baseline_identity,
                "operand_form": (
                    abi.WIDENING_PRODUCT_REDUCE_DEQUANTIZE_F32_PACKED_I4_OPERAND_FORM
                ),
                "packing_layout": (
                    abi.WIDENING_PRODUCT_REDUCE_DEQUANTIZE_F32_PACKED_I4_PACKING_LAYOUT
                ),
                "unpack_intent": (
                    abi.WIDENING_PRODUCT_REDUCE_DEQUANTIZE_F32_PACKED_I4_UNPACK_INTENT
                ),
                "runtime_n_unit": "packed input bytes",
            }
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
                uses_packed_i4_resource=uses_packed_i4_resource,
            ),
            encoding="utf-8",
        )
        evidence["measurement_harness"] = {
            "path": str(harness_path),
            "sha256": abi.sha256_file(harness_path),
            "correctness_before_timing": True,
            "baseline_identity": baseline_identity,
            "generated_function": expectation.function_name,
            "packed_i4_reference_oracle": uses_packed_i4_resource,
        }

        remote: dict[str, Any] | None = None
        if args.dry_run:
            evidence["ssh_evidence"] = False
            evidence["status"] = "dry_run_success"
            result_classification = not_measured_result_classification(
                "dry-run generated and validated bundle/harness but did not run ssh rvv timing"
            )
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
            result_classification = classify_parsed_timing(
                remote.get("parsed_timing", {})
            )
        source_record_context = source_backed_pressure_profile_record_context(
            generation_result=generation_result,
            expectation=expectation,
            object_path=object_path,
            header_path=header_path,
            config=config,
            result_classification=result_classification,
        )
        provider_feedback_tie_back = packed_i4_provider_feedback_tie_back(
            generation_result=generation_result,
            expectation=expectation,
            uses_packed_i4_resource=uses_packed_i4_resource,
            result_classification=result_classification,
            measurement_evidence_id=(
                f"{run_id}/{expectation.kind}/same_target_measurement_evidence.json"
            ),
            source_record_context=source_record_context,
        )
        if uses_packed_i4_resource:
            fields = provider_feedback_tie_back["fields"]
            schedule_decision_evidence = {
                "schedule_decision": fields["schedule_decision"],
                "provider_schedule_decision_contract": fields[
                    "schedule_decision_contract"
                ],
                "provider_schedule_decision": fields["schedule_decision"],
                "provider_schedule_decision_reason": fields[
                    "schedule_decision_reason"
                ],
                "provider_resource_cost_contract": fields[
                    "resource_cost_contract"
                ],
                "provider_resource_cost_model": fields["resource_cost_model"],
                "provider_resource_cost_loop_body_steps": packed_i4_resource_int(
                    fields, "resource_cost_loop_body_steps"
                ),
                "provider_resource_cost_blocker": fields[
                    "resource_cost_blocker"
                ],
                "provider_performance_admission_decision": fields[
                    "performance_admission_decision"
                ],
                "provider_realization_admission_schedule_decision_contract": fields[
                    "realization_admission_schedule_decision_contract"
                ],
                "provider_realization_admission_schedule_decision": fields[
                    "realization_admission_schedule_decision"
                ],
                "provider_realization_admission_schedule_decision_reason": fields[
                    "realization_admission_schedule_decision_reason"
                ],
                "source": (
                    "provider-owned packed-i4 schedule decision mirrored by "
                    "validated generated object/header metadata"
                ),
            }
            evidence["measurement_schedule_decision_evidence"] = (
                schedule_decision_evidence
            )
            evidence["measurement_harness"].update(
                {
                    key: value
                    for key, value in schedule_decision_evidence.items()
                    if key != "source"
                }
            )
            evidence.setdefault("packed_i4_reference_oracle", {}).update(
                {
                    key: value
                    for key, value in schedule_decision_evidence.items()
                    if key != "source"
                }
            )
        evidence["result_classification"] = result_classification
        evidence["provider_feedback_tie_back"] = provider_feedback_tie_back
        maturity_input = provider_feedback_tie_back.get(
            "maturity_contract_evidence_input"
        )
        if maturity_input:
            evidence["performance_maturity_contract_evidence_input"] = (
                maturity_input
            )
        same_target_record = provider_feedback_tie_back.get(
            "same_target_measurement_record"
        )
        if same_target_record:
            evidence["same_target_measurement_record"] = same_target_record
        evidence["op_summary"] = op_measurement_summary(
            expectation=expectation,
            generation_result=generation_result,
            harness_path=harness_path,
            config=config,
            remote=remote,
            uses_packed_i4_resource=uses_packed_i4_resource,
            result_classification=result_classification,
            provider_feedback_tie_back=provider_feedback_tie_back,
        )
        evidence["completed_at"] = abi.utc_timestamp()
        abi.write_json(
            op_artifact_dir / "same_target_measurement_evidence.json", evidence
        )
        return evidence
    except Exception as exc:  # noqa: BLE001 - evidence should record blockers.
        evidence["status"] = "blocked" if not args.dry_run else "failed"
        evidence["completed_at"] = abi.utc_timestamp()
        evidence["diagnostic"] = abi.sanitize_text(exc)
        op_artifact_dir.mkdir(parents=True, exist_ok=True)
        abi.write_json(
            op_artifact_dir / "same_target_measurement_evidence.json", evidence
        )
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
    artifact_dir = abi.prepare_artifact_dir(
        args.artifact_root, run_id, args.overwrite
    )
    config = MeasurementConfig(
        counts=args.measure_count or list(DEFAULT_MEASURE_COUNTS),
        dequant_scale_values=args.dequant_scale
        or list(abi.DEFAULT_DEQUANT_SCALE_VALUES),
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
            "packed_i4_baseline_identities": PACKED_I4_BASELINE_IDENTITIES,
        },
        "op_results": {},
        "performance_maturity_contract_inputs": {},
        "same_target_measurement_records": {},
    }
    try:
        validate_measurement_config(config)
        op_kinds = args.op_kind or list(DEFAULT_OP_KINDS)
        if len(set(op_kinds)) != len(op_kinds):
            raise abi.EvidenceError(
                f"duplicate --op-kind values are not allowed: {op_kinds}"
            )
        if args.input is not None and len(op_kinds) != 1:
            raise abi.EvidenceError(
                "--input may only be used with exactly one --op-kind"
            )
        expectations = [
            selected_pre_realized_expectation(op_kind, args.input)
            for op_kind in op_kinds
        ]
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
            maturity_input = result.get(
                "performance_maturity_contract_evidence_input"
            )
            if maturity_input:
                evidence["performance_maturity_contract_inputs"][
                    expectation.kind
                ] = maturity_input
            same_target_record = result.get("same_target_measurement_record")
            if same_target_record:
                evidence["same_target_measurement_records"][
                    expectation.kind
                ] = same_target_record

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
                classification = result.get("result_classification", {})
                maturity_input = result.get(
                    "performance_maturity_contract_evidence_input", {}
                )
                print(
                    f"[{op_kind}] summaries={parsed.get('summary_record_count', 0)} "
                    f"measurements={parsed.get('measurement_record_count', 0)} "
                    f"classification={classification.get('classification', '')} "
                    f"best_speedup_range={classification.get('best_speedup_range', '')} "
                    "selection_eligible="
                    f"{maturity_input.get('provider_performance_selection_eligible', '')} "
                    f"claim_allowed={maturity_input.get('performance_win_claim_allowed', '')}"
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
        if (
            op_kind == "widening_product_reduce_dequantize_f32"
            and "packed_i4_reference_oracle" in harness
        ):
            raise AssertionError(
                "self-test default dequant measurement harness leaked packed-i4 oracle"
            )
    packed_expectation = selected_pre_realized_expectation(
        "widening_product_reduce_dequantize_f32"
    )
    packed_harness = measurement_harness_source(
        header_file_name="artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h",
        expectation=packed_expectation,
        config=config,
        compile_flags_summary=shlex.join(config.compile_flags),
        uses_packed_i4_resource=True,
    )
    for token in [
        PACKED_I4_BASELINE_IDENTITIES["widening_product_reduce_dequantize_f32"],
        "baseline_product_reduction_dequant_packed_i4_v1",
        "pack_signed_i4_pair",
        "sign_extend_i4",
        "low_product + high_product",
        "packed_i4_reference_oracle=true",
        "runtime_n_unit=packed_bytes",
        "PASS op=widening_product_reduce_dequantize_f32 measurement",
    ]:
        if token not in packed_harness:
            raise AssertionError(
                f"self-test packed-i4 measurement harness lost token: {token}"
            )
    packed_clamp_expectation = selected_pre_realized_expectation(
        "widening_product_reduce_dequant_clamp_f32"
    )
    packed_clamp_harness = measurement_harness_source(
        header_file_name="artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h",
        expectation=packed_clamp_expectation,
        config=config,
        compile_flags_summary=shlex.join(config.compile_flags),
        uses_packed_i4_resource=True,
    )
    for token in [
        PACKED_I4_BASELINE_IDENTITIES[
            "widening_product_reduce_dequant_clamp_f32"
        ],
        "baseline_product_reduction_dequant_clamp_packed_i4_v1",
        "pack_signed_i4_pair",
        "sign_extend_i4",
        "low_product + high_product",
        "bound_pairs=",
        "packed_i4_reference_oracle=true",
        "runtime_n_unit=packed_bytes",
        "PASS op=widening_product_reduce_dequant_clamp_f32 measurement",
    ]:
        if token not in packed_clamp_harness:
            raise AssertionError(
                "self-test packed-i4 clamp measurement harness lost token: "
                f"{token}"
            )
    parsed = parse_measurement_stdout(
        "\n".join(
            [
                "MEASURE_CONFIG op=widening_product_reduce_dequantize_f32",
                "CORRECTNESS op=widening_product_reduce_dequantize_f32 n=257",
                "MEASURE op=widening_product_reduce_dequantize_f32 n=257 "
                "baseline_ns=100 generated_ns=50",
                "SUMMARY op=widening_product_reduce_dequantize_f32 n=257 "
                "pattern=0 scale=-0.125 baseline_best_per_iter_ns=100 "
                "generated_best_per_iter_ns=125 best_speedup=0.800000",
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
    regression = classify_parsed_timing(parsed)
    if (
        regression["classification"] != RESULT_CLASSIFICATION_REGRESSION
        or regression["outcome_family"] != RESULT_CLASSIFICATION_NO_WIN
        or regression["best_speedup_range"] != "0.800000..0.800000"
    ):
        raise AssertionError("self-test timing classifier lost regression result")
    win = classify_parsed_timing(
        parse_measurement_stdout(
            "SUMMARY op=widening_product_reduce_dequantize_f32 n=257 "
            "pattern=0 scale=-0.125 baseline_best_per_iter_ns=200 "
            "generated_best_per_iter_ns=100 best_speedup=2.000000"
        )
    )
    if win["classification"] != RESULT_CLASSIFICATION_WIN:
        raise AssertionError("self-test timing classifier lost win result")
    mixed = classify_parsed_timing(
        parse_measurement_stdout(
            "\n".join(
                [
                    "SUMMARY op=widening_product_reduce_dequantize_f32 n=257 "
                    "pattern=0 scale=-0.125 baseline_best_per_iter_ns=100 "
                    "generated_best_per_iter_ns=100 best_speedup=1.000000",
                    "SUMMARY op=widening_product_reduce_dequantize_f32 n=4096 "
                    "pattern=1 scale=0.375 baseline_best_per_iter_ns=110 "
                    "generated_best_per_iter_ns=100 best_speedup=1.100000",
                ]
            )
        )
    )
    if mixed["classification"] != RESULT_CLASSIFICATION_NO_WIN:
        raise AssertionError("self-test timing classifier lost no-win result")
    feedback_metadata = abi.expected_low_precision_resource_metadata(
        packed_expectation, packed_i4=True
    )

    def self_test_source_record_context(
        expectation: abi.OpExpectation,
        result_classification: dict[str, Any],
    ) -> dict[str, Any]:
        measured = (
            result_classification["classification"]
            != RESULT_CLASSIFICATION_NOT_MEASURED
        )
        return {
            "source_record_contract": SOURCE_BACKED_MEASUREMENT_RECORD_CONTRACT,
            "source_selected_variant": expectation.selected_variant,
            "source_selected_input": str(expectation.input_path),
            "source_generated_function": expectation.function_name,
            "generated_artifact_identity_contract": (
                GENERATED_ARTIFACT_IDENTITY_CONTRACT
            ),
            "generated_artifact_object_path": (
                f"self-test/generated_bundle/{expectation.kind}.o"
            ),
            "generated_artifact_object_sha256": (
                f"self-test-object-sha256-{expectation.kind}"
            ),
            "generated_artifact_header_path": (
                f"self-test/generated_bundle/{expectation.kind}.h"
            ),
            "generated_artifact_header_sha256": (
                f"self-test-header-sha256-{expectation.kind}"
            ),
            "measurement_target": (
                PACKED_I4_SSH_TARGET_PROFILE if measured else ""
            ),
            "measurement_target_provenance": MEASUREMENT_TARGET_PROVENANCE,
            "measurement_runtime_count_set": runtime_count_set(config.counts),
            "measurement_runtime_count_provenance": (
                MEASUREMENT_RUNTIME_COUNT_PROVENANCE
            ),
            "pressure_profile_label": PRODUCTION_PRESSURE_PROFILE_LABEL,
            "pressure_profile_label_provenance": (
                PRODUCTION_PRESSURE_PROFILE_LABEL_PROVENANCE
            ),
        }

    def check_packed_i4_contract_input(
        result_classification: dict[str, Any],
        *,
        expected_alignment: str,
        expected_denial_reason: str,
        expected_update_required: bool,
    ) -> dict[str, Any]:
        measurement_evidence_id = (
            "self-test/"
            f"{result_classification['classification']}/same_target_measurement_evidence.json"
        )
        source_record_context = self_test_source_record_context(
            packed_expectation, result_classification
        )
        tie_back = packed_i4_provider_feedback_tie_back(
            generation_result={
                "widening_product_reduction_boundary": {
                    "route_metadata": feedback_metadata
                }
            },
            expectation=packed_expectation,
            uses_packed_i4_resource=True,
            result_classification=result_classification,
            measurement_evidence_id=measurement_evidence_id,
            source_record_context=source_record_context,
        )
        maturity_input = tie_back["maturity_contract_evidence_input"]
        if tie_back["result_alignment"] != expected_alignment:
            raise AssertionError("self-test packed-i4 contract alignment changed")
        if (
            tie_back["performance_win_claim_allowed"]
            or maturity_input["performance_win_claim_allowed"]
        ):
            raise AssertionError("self-test packed-i4 contract allowed stale win")
        record = tie_back["same_target_measurement_record"]
        if tuple(record.keys()) != PACKED_I4_SAME_TARGET_MEASUREMENT_RECORD_FIELDS:
            raise AssertionError(
                "self-test packed-i4 measurement record field order changed"
            )
        expected_record = {
            field: maturity_input[field]
            for field in PACKED_I4_SAME_TARGET_MEASUREMENT_RECORD_FIELDS
        }
        if record != expected_record:
            raise AssertionError(
                "self-test packed-i4 measurement record lost evidence fields"
            )
        for reporting_only_field in (
            "contract_alignment",
            "provider_remediation_plan",
            "provider_remediation_product_plan",
        ):
            if reporting_only_field in record:
                raise AssertionError(
                    "self-test packed-i4 measurement record leaked "
                    f"{reporting_only_field}"
                )
        if maturity_input["measurement_evidence_id"] != measurement_evidence_id:
            raise AssertionError("self-test packed-i4 measurement evidence id lost")
        if maturity_input["authority"] != PACKED_I4_MATURITY_CONTRACT_AUTHORITY:
            raise AssertionError("self-test packed-i4 authority contract changed")
        expected_measured = (
            result_classification["classification"]
            != RESULT_CLASSIFICATION_NOT_MEASURED
        )
        if maturity_input["same_target_measurement"] != expected_measured:
            raise AssertionError("self-test packed-i4 same-target flag changed")
        if maturity_input["ssh_evidence"] != expected_measured:
            raise AssertionError("self-test packed-i4 ssh evidence flag changed")
        expected_target_profile = (
            PACKED_I4_SSH_TARGET_PROFILE if expected_measured else ""
        )
        if maturity_input["target_profile"] != expected_target_profile:
            raise AssertionError("self-test packed-i4 target profile changed")
        if (
            record["source_record_contract"]
            != SOURCE_BACKED_MEASUREMENT_RECORD_CONTRACT
            or record["source_selected_variant"]
            != packed_expectation.selected_variant
            or record["source_generated_function"]
            != packed_expectation.function_name
            or record["generated_artifact_identity_contract"]
            != GENERATED_ARTIFACT_IDENTITY_CONTRACT
            or record["measurement_runtime_count_set"]
            != runtime_count_set(config.counts)
            or record["pressure_profile_label"]
            != PRODUCTION_PRESSURE_PROFILE_LABEL
            or record["pressure_profile_label_provenance"]
            != PRODUCTION_PRESSURE_PROFILE_LABEL_PROVENANCE
        ):
            raise AssertionError(
                "self-test packed-i4 source-backed measurement record lost "
                "selected-boundary, artifact, runtime-count, or pressure "
                "provenance"
            )
        if (
            maturity_input["performance_preference_denial_reason"]
            != expected_denial_reason
        ):
            raise AssertionError("self-test packed-i4 denial reason changed")
        if (
            maturity_input["provider_contract_update_required"]
            != expected_update_required
        ):
            raise AssertionError("self-test packed-i4 update-required flag changed")
        if (
            not maturity_input["correctness_execution_allowed"]
            or not maturity_input["performance_preference_denied"]
        ):
            raise AssertionError(
                "self-test packed-i4 contract lost correctness/performance split"
            )
        return tie_back

    clamp_feedback_metadata = abi.expected_low_precision_resource_metadata(
        packed_clamp_expectation, packed_i4=True
    )
    clamp_tie_back = packed_i4_provider_feedback_tie_back(
        generation_result={
            "widening_product_reduction_boundary": {
                "route_metadata": clamp_feedback_metadata
            }
        },
        expectation=packed_clamp_expectation,
        uses_packed_i4_resource=True,
        result_classification=not_measured_result_classification(
            "self-test clamp not measured"
        ),
        measurement_evidence_id=(
            "self-test/clamp/same_target_measurement_evidence.json"
        ),
        source_record_context=self_test_source_record_context(
            packed_clamp_expectation,
            not_measured_result_classification("self-test clamp not measured"),
        ),
    )
    clamp_fields = clamp_tie_back["fields"]
    if (
        clamp_fields["selected_candidate"]
        != abi.WIDENING_PRODUCT_REDUCE_DEQUANT_CLAMP_F32_PACKED_I4_RESOURCE_SELECTED_CANDIDATE
        or clamp_fields["performance_baseline"]
        != abi.WIDENING_PRODUCT_REDUCE_DEQUANT_CLAMP_F32_PACKED_I4_PERFORMANCE_BASELINE
        or clamp_fields["runtime_abi_order"]
        != "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"
        or clamp_tie_back["maturity_contract_evidence_input"][
            "provider_runtime_abi_order"
        ]
        != "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"
    ):
        raise AssertionError(
            "self-test packed-i4 clamp provider tie-back lost selected "
            "candidate, baseline, or clamp ABI order"
        )

    not_measured = not_measured_result_classification("self-test-not-measured")
    check_packed_i4_contract_input(
        not_measured,
        expected_alignment="not-measured",
        expected_denial_reason="same-target-measurement-not-run",
        expected_update_required=False,
    )
    regression_tie_back = check_packed_i4_contract_input(
        regression,
        expected_alignment="same-no-win-family-denies-performance-preference",
        expected_denial_reason="same-target-measurement-no-win-or-regression",
        expected_update_required=True,
    )
    check_packed_i4_contract_input(
        mixed,
        expected_alignment="matches-provider-maturity-outcome",
        expected_denial_reason="same-target-measurement-no-win-or-regression",
        expected_update_required=False,
    )
    check_packed_i4_contract_input(
        win,
        expected_alignment=(
            "measurement-win-conflicts-with-provider-maturity-contract-requires-provider-update"
        ),
        expected_denial_reason="provider-contract-performance-selection-ineligible",
        expected_update_required=True,
    )

    def expect_maturity_input_failure(
        field: str, stale_value: Any, expected_token: str
    ) -> None:
        fields = regression_tie_back["fields"]
        result_classification = regression
        measurement_evidence_id = (
            "self-test/"
            f"{result_classification['classification']}/same_target_measurement_evidence.json"
        )
        source_record_context = self_test_source_record_context(
            packed_expectation, result_classification
        )
        stale_input = dict(
            regression_tie_back["maturity_contract_evidence_input"]
        )
        stale_input[field] = stale_value
        try:
            validate_packed_i4_maturity_contract_evidence_input(
                fields=fields,
                result_classification=result_classification,
                measurement_evidence_id=measurement_evidence_id,
                source_record_context=source_record_context,
                maturity_input=stale_input,
                context="self-test stale packed-i4 maturity input",
            )
        except abi.EvidenceError as exc:
            if expected_token not in str(exc):
                raise AssertionError(
                    "self-test stale packed-i4 maturity input failure "
                    f"for {field} missed token {expected_token}: {exc}"
                ) from exc
            return
        raise AssertionError(
            f"self-test stale packed-i4 maturity input accepted {field}"
        )

    for field, stale_value, expected_token in [
        (
            "measurement_evidence_id",
            "stale/same_target_measurement_evidence.json",
            "measurement_evidence_id",
        ),
        ("authority", "metadata-derived-policy-authority", "authority"),
        ("measurement_classification", RESULT_CLASSIFICATION_WIN, "classification"),
        ("measurement_outcome_family", RESULT_CLASSIFICATION_WIN, "outcome_family"),
        ("measurement_best_speedup_range", "2.000000..2.500000", "speedup"),
        ("correctness_record_count", 0, "correctness_record_count"),
        ("same_target_measurement", False, "same_target_measurement"),
        ("ssh_evidence", False, "ssh_evidence"),
        ("target_profile", "local-x86", "target_profile"),
        (
            "source_selected_variant",
            "metadata-only-selected-variant",
            "source_selected_variant",
        ),
        (
            "generated_artifact_object_sha256",
            "metadata-only-object-sha256",
            "generated_artifact_object_sha256",
        ),
        ("measurement_target", "local-x86", "measurement_target"),
        (
            "measurement_runtime_count_set",
            "257",
            "measurement_runtime_count_set",
        ),
        (
            "pressure_profile_label",
            "q8-label-only-pressure",
            "pressure_profile_label",
        ),
        ("provider_maturity_outcome", RESULT_CLASSIFICATION_WIN, "maturity"),
        (
            "provider_resource_planning_contract",
            "metadata-derived-resource-planning-contract",
            "planning_contract",
        ),
        (
            "provider_resource_operand_form",
            "metadata-only-packed-form",
            "operand_form",
        ),
        (
            "provider_resource_source_signedness",
            "unsigned",
            "source_signedness",
        ),
        (
            "provider_resource_storage_element_width",
            16,
            "storage_element_width",
        ),
        (
            "provider_resource_effective_element_width",
            8,
            "effective_element_width",
        ),
        (
            "provider_resource_packing_layout",
            "metadata-only-packed-layout",
            "packing_layout",
        ),
        (
            "provider_resource_unpack_intent",
            "metadata-only-unpack-intent",
            "unpack_intent",
        ),
        (
            "provider_resource_vsetvl_region_count",
            3,
            "vsetvl_region_count",
        ),
        (
            "provider_resource_cost_contract",
            "metadata-derived-resource-cost-contract",
            "resource_cost_contract",
        ),
        (
            "provider_resource_cost_model",
            "metadata-derived-resource-cost-model",
            "resource_cost_model",
        ),
        (
            "provider_resource_cost_loop_body_steps",
            99,
            "resource_cost_loop_body_steps",
        ),
        (
            "provider_resource_cost_blocker",
            "metadata-derived-resource-cost-blocker",
            "resource_cost_blocker",
        ),
        (
            "provider_performance_admission_decision",
            "metadata-derived-performance-admission",
            "performance_admission_decision",
        ),
        (
            "provider_runtime_avl_source",
            "metadata-derived-avl",
            "runtime_avl_source",
        ),
        (
            "provider_resource_route_family_plan",
            "stale-route-family-plan.v1",
            "route_family_plan",
        ),
        (
            "provider_supported_mirror",
            "provider_supported_mirror:stale",
            "provider_supported_mirror",
        ),
        (
            "provider_runtime_abi_order",
            "lhs,rhs,out,n",
            "runtime_abi_order",
        ),
        (
            "provider_primitive_chain_kind",
            "stale-primitive-chain-kind",
            "primitive_chain_kind",
        ),
        (
            "provider_primitive_source_extension",
            "stale-primitive-source-extension",
            "primitive_source_extension",
        ),
        (
            "provider_primitive_reduction_intrinsic",
            "__riscv_vwredsum_vs_i32m1_i32m1",
            "primitive_reduction_intrinsic",
        ),
        (
            "provider_remediation_handoff_contract",
            "stale-remediation-handoff.v1",
            "remediation_handoff_contract",
        ),
        (
            "provider_remediation_measurement_evidence",
            "stale/remediation/same_target_measurement_evidence.json",
            "remediation_measurement_evidence",
        ),
        (
            "provider_remediation_decision",
            "stale-remediation-decision",
            "remediation_decision",
        ),
        (
            "provider_remediation_dispatch_preference",
            PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH,
            "remediation_dispatch_preference",
        ),
        (
            "provider_remediation_product_plan",
            "metadata-only-packed-i4-product-plan",
            "remediation_product_plan",
        ),
        (
            "provider_remediation_vl_plan",
            "metadata-only-packed-i4-vl-plan",
            "remediation_vl_plan",
        ),
        (
            "provider_schedule_decision",
            "metadata-only-packed-i4-schedule-decision",
            "schedule_decision",
        ),
        (
            "provider_realization_admission_evidence",
            "sibling-route-packed-i4-measurement-evidence",
            "realization_admission_evidence",
        ),
        (
            "provider_realization_admission_schedule_decision",
            "metadata-only-packed-i4-admission-schedule-decision",
            "realization_admission_schedule_decision",
        ),
        (
            "target_capability_legality_mirror",
            "stale-target-capability-legality",
            "target_capability_legality_mirror",
        ),
        (
            "provider_performance_selection_eligible",
            "true",
            "selection",
        ),
        (
            "provider_dispatch_preference",
            PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH,
            "dispatch",
        ),
        ("performance_win_claim_allowed", True, "performance_win_claim_allowed"),
    ]:
        expect_maturity_input_failure(field, stale_value, expected_token)

    def expect_stale_provider_metadata_failure(
        metadata_key: str, stale_value: str, expected_token: str
    ) -> None:
        stale_metadata = dict(feedback_metadata)
        stale_metadata[metadata_key] = stale_value
        try:
            packed_i4_provider_feedback_tie_back(
                generation_result={
                    "widening_product_reduction_boundary": {
                        "route_metadata": stale_metadata
                    }
                },
                expectation=packed_expectation,
                uses_packed_i4_resource=True,
                result_classification=regression,
                measurement_evidence_id=(
                    "self-test/stale-provider/same_target_measurement_evidence.json"
                ),
                source_record_context=self_test_source_record_context(
                    packed_expectation, regression
                ),
            )
        except abi.EvidenceError as exc:
            if expected_token not in str(exc):
                raise AssertionError(
                    "self-test stale packed-i4 provider metadata failure for "
                    f"{metadata_key} missed token {expected_token}: {exc}"
                ) from exc
            return
        raise AssertionError(
            f"self-test stale packed-i4 provider metadata accepted {metadata_key}"
        )

    def expect_missing_provider_metadata_failure(
        metadata_key: str, expected_token: str
    ) -> None:
        missing_metadata = dict(feedback_metadata)
        del missing_metadata[metadata_key]
        try:
            packed_i4_provider_feedback_tie_back(
                generation_result={
                    "widening_product_reduction_boundary": {
                        "route_metadata": missing_metadata
                    }
                },
                expectation=packed_expectation,
                uses_packed_i4_resource=True,
                result_classification=regression,
                measurement_evidence_id=(
                    "self-test/missing-provider/same_target_measurement_evidence.json"
                ),
                source_record_context=self_test_source_record_context(
                    packed_expectation, regression
                ),
            )
        except abi.EvidenceError as exc:
            if expected_token not in str(exc):
                raise AssertionError(
                    "self-test missing packed-i4 provider metadata failure for "
                    f"{metadata_key} missed token {expected_token}: {exc}"
                ) from exc
            return
        raise AssertionError(
            f"self-test missing packed-i4 provider metadata accepted {metadata_key}"
        )

    for metadata_key, stale_value, expected_token in [
        (
            "tcrv_rvv.low_precision_resource.planning_contract",
            "metadata-derived-resource-planning-contract",
            "planning_contract",
        ),
        (
            "tcrv_rvv.low_precision_resource.operand_form",
            "metadata-only-packed-form",
            "operand_form",
        ),
        (
            "tcrv_rvv.low_precision_resource.source_signedness",
            "unsigned",
            "source_signedness",
        ),
        (
            "tcrv_rvv.low_precision_resource.storage_element_width",
            "16",
            "storage_element_width",
        ),
        (
            "tcrv_rvv.low_precision_resource.effective_element_width",
            "8",
            "effective_element_width",
        ),
        (
            "tcrv_rvv.low_precision_resource.packing_layout",
            "metadata-only-packed-layout",
            "packing_layout",
        ),
        (
            "tcrv_rvv.low_precision_resource.unpack_intent",
            "metadata-only-unpack-intent",
            "unpack_intent",
        ),
        (
            "tcrv_rvv.low_precision_resource.vsetvl_region_count",
            "3",
            "vsetvl_region_count",
        ),
        (
            "tcrv_rvv.low_precision_resource.runtime_avl_source",
            "metadata-derived-avl",
            "runtime_avl_source",
        ),
        (
            "tcrv_rvv.low_precision_resource.route_family_plan",
            "stale-route-family-plan.v1",
            "route_family_plan",
        ),
        (
            "tcrv_rvv.low_precision_resource.provider_supported_mirror",
            "provider_supported_mirror:stale",
            "provider_supported_mirror",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_handoff_contract",
            "stale-remediation-handoff.v1",
            "remediation_handoff_contract",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_measurement_evidence",
            "stale/remediation/same_target_measurement_evidence.json",
            "remediation_measurement_evidence",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_decision",
            "stale-remediation-decision",
            "remediation_decision",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_product_plan",
            "metadata-only-packed-i4-product-plan",
            "remediation_product_plan",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_vl_plan",
            "metadata-only-packed-i4-vl-plan",
            "remediation_vl_plan",
        ),
        (
            "tcrv_rvv.low_precision_resource.schedule_decision",
            "metadata-only-packed-i4-schedule-decision",
            "schedule_decision",
        ),
        (
            "tcrv_rvv.low_precision_resource.performance_maturity_outcome",
            RESULT_CLASSIFICATION_WIN,
            "performance_maturity_outcome",
        ),
        (
            "tcrv_rvv.low_precision_resource.performance_selection_eligible",
            "true",
            "performance_selection_eligible",
        ),
        (
            "tcrv_rvv.low_precision_resource.dispatch_preference",
            PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH,
            "dispatch_preference",
        ),
        (
            "tcrv_rvv.low_precision_resource.performance_best_speedup_range",
            "2.000000..2.500000",
            "performance_best_speedup_range",
        ),
        (
            "tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic",
            "__riscv_vwredsum_vs_i32m1_i32m1",
            "primitive_reduction_intrinsic",
        ),
        (
            "tcrv_rvv.low_precision_resource.primitive_chain_kind",
            "stale-primitive-chain-kind",
            "primitive_chain_kind",
        ),
        (
            "tcrv_rvv.low_precision_resource.primitive_source_extension",
            "stale-primitive-source-extension",
            "primitive_source_extension",
        ),
        (
            "tcrv_rvv.low_precision_resource.realization_decision",
            "stale-realization-decision",
            "realization_decision",
        ),
        (
            "tcrv_rvv.low_precision_resource.target_capability_legality_mirror",
            "stale-target-capability-legality",
            "target_capability_legality_mirror",
        ),
    ]:
        expect_stale_provider_metadata_failure(
            metadata_key, stale_value, expected_token
        )
    for metadata_key, expected_token in [
        (
            "tcrv_rvv.low_precision_resource.route_family_plan",
            "route_family_plan",
        ),
        (
            "tcrv_rvv.low_precision_resource.provider_supported_mirror",
            "provider_supported_mirror",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_handoff_contract",
            "remediation_handoff_contract",
        ),
        (
            "tcrv_rvv.low_precision_resource.remediation_product_plan",
            "remediation_product_plan",
        ),
    ]:
        expect_missing_provider_metadata_failure(metadata_key, expected_token)
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
        "--input",
        type=Path,
        default=None,
        help=(
            "override the pre-realized selected-body MLIR fixture for exactly "
            "one --op-kind; packed-i4 timing support is selected only after "
            "provider-owned low-precision resource metadata validates the "
            "generated bundle"
        ),
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
