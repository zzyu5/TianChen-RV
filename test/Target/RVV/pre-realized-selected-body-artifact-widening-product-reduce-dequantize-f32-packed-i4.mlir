// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/resource_decision = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"/resource_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/region_count = 2 : i64/region_count = 3 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-REGION-COUNT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/from_phase = "load-product-reduce"/from_phase = "tail-product-reduce"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-FROM-PHASE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"/s//tcrv_rvv.low_precision_resource.realization_decision", value = "artifact-name-derived-resource-decision"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REALIZATION-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"/s//tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-performance-win.v1"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-FEEDBACK
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "false"/s//tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "true"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-SELECTION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-rvv-emitc-to-cpp | FileCheck %s --check-prefix=CPP

// Focused Gate 4 fixture for the same product-reduction-dequant op kind with
// an explicit signed packed-i4 selected resource. The candidate is authority
// only because it is carried in the typed pre-realized tcrv_rvv body and then
// consumed by RVV selected-body realization/provider planning.

module {
  tcrv.exec.kernel @pre_realized_body_product_reduce_dequantize_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// REALIZED-DAG: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.operand_form = "packed-i4-nibbles"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.storage_element_width = 8 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.effective_element_width = 4 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packing_layout = "two-signed-i4-elements-per-byte-low-high-nibbles"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.unpack_intent = "sign-extend-i4-nibbles-before-widening-product"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_unroll_factor = 1 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_feedback = "same-target-packed-i4-no-win.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_baseline = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_best_speedup_range = "0.688427..0.705724"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_action = "no-win-repair-required-before-performance-claim"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity = "executable-not-performance-mature"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity_evidence = "same-target-packed-i4-product-pair-sum-regression-gate6.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity_outcome = "regression"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_selection_eligible = "false"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.dispatch_preference = "not-performance-preferred"
// REALIZED: tcrv_rvv.vsetvl_region_marker
// REALIZED-SAME: phase = "load-product-reduce"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 1 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"
// REALIZED: tcrv_rvv.gearbox_cross_region_handoff
// REALIZED-SAME: from_phase = "load-product-reduce"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"
// REALIZED: tcrv_rvv.vsetvl_region_marker
// REALIZED-SAME: phase = "dequant-store"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 2 : i64

// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realized_vsetvl_region_count", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_region_index", value = "1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_region_index", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_phase", value = "load-product-reduce"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_phase", value = "dequant-store"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_best_speedup_range", value = "0.688427..0.705724"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_action", value = "no-win-repair-required-before-performance-claim"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity", value = "executable-not-performance-mature"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity_evidence", value = "same-target-packed-i4-product-pair-sum-regression-gate6.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity_outcome", value = "regression"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "false"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"}

// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]
// HEADER: tianchenrv.rvv.low_precision_resource.operand_form: packed-i4-nibbles
// HEADER: tianchenrv.rvv.low_precision_resource.source_signedness: signed
// HEADER: tianchenrv.rvv.low_precision_resource.storage_element_width: 8
// HEADER: tianchenrv.rvv.low_precision_resource.effective_element_width: 4
// HEADER: tianchenrv.rvv.low_precision_resource.packing_layout: two-signed-i4-elements-per-byte-low-high-nibbles
// HEADER: tianchenrv.rvv.low_precision_resource.unpack_intent: sign-extend-i4-nibbles-before-widening-product
// HEADER: tianchenrv.rvv.low_precision_resource.peak_live_vector_groups: 7
// HEADER: tianchenrv.rvv.low_precision_resource.realization_decision: consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.realized_vsetvl_region_count: 2
// HEADER: tianchenrv.rvv.low_precision_resource.product_region_index: 1
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_region_index: 2
// HEADER: tianchenrv.rvv.low_precision_resource.product_phase: load-product-reduce
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_phase: dequant-store
// HEADER: tianchenrv.rvv.low_precision_resource.performance_feedback: same-target-packed-i4-no-win.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_baseline: scalar-c-reference/product-reduction-dequant-packed-i4-v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_best_speedup_range: 0.688427..0.705724
// HEADER: tianchenrv.rvv.low_precision_resource.performance_action: no-win-repair-required-before-performance-claim
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity: executable-not-performance-mature
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity_evidence: same-target-packed-i4-product-pair-sum-regression-gate6.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity_outcome: regression
// HEADER: tianchenrv.rvv.low_precision_resource.performance_selection_eligible: false
// HEADER: tianchenrv.rvv.low_precision_resource.dispatch_preference: not-performance-preferred
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vwmul_vv_i16mf2
// CPP: __riscv_vwmul_vv_i16mf2
// CPP: __riscv_vadd_vv_i16mf2
// CPP: __riscv_vwredsum_vs_i16mf2_i32m1
// CPP: tcrv_emitc.assign target=dot_acc_vec

// STALE-PACKED-DECISION: selected-body realization low-precision direct-contraction structure has stale or inconsistent vsetvl region marker
// STALE-PACKED-DECISION-SAME: resource decision 'consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1'

// STALE-PACKED-REGION-COUNT: requires region_count to match the bounded Gearbox resource decision

// STALE-PACKED-FROM-PHASE: requires from_phase 'load-product-reduce'

// STALE-ARTIFACT-REALIZATION-DECISION: low_precision_resource.realization_decision provenance must mirror provider-selected low-precision direct-contraction resource realization decision 'consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1'
// STALE-ARTIFACT-REALIZATION-DECISION-SAME: artifact-name-derived-resource-decision

// STALE-ARTIFACT-PERFORMANCE-FEEDBACK: low_precision_resource.performance_feedback provenance must mirror provider-selected low-precision direct-contraction resource performance feedback 'same-target-packed-i4-no-win.v1'
// STALE-ARTIFACT-PERFORMANCE-FEEDBACK-SAME: same-target-packed-i4-performance-win.v1

// STALE-ARTIFACT-PERFORMANCE-SELECTION: low_precision_resource.performance_selection_eligible provenance must mirror provider-selected low-precision direct-contraction resource performance selection eligibility 'false'
// STALE-ARTIFACT-PERFORMANCE-SELECTION-SAME: true
