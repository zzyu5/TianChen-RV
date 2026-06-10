// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/resource_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"/resource_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/region_count = 2 : i64/region_count = 3 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-REGION-COUNT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/from_phase = "load-product-reduce"/from_phase = "tail-product-reduce"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-FROM-PHASE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '/gearbox_cross_region_handoff/s/remediation_statement_strategy = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"/remediation_statement_strategy = "metadata-only-packed-i4-unpack-plan"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-REMEDIATION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '/gearbox_cross_region_handoff/s/remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1"/remediation_product_plan = "metadata-only-packed-i4-product-plan"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-REMEDIATION-PRODUCT-PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '/gearbox_cross_region_handoff/s/schedule_decision = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"/schedule_decision = "metadata-only-packed-i4-schedule-decision"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-SCHEDULE-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '/gearbox_cross_region_handoff/s/remediation_vector_budget = "packed-i4-remediation-budget-5of32-vector-groups", //' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-PACKED-HANDOFF-REMEDIATION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"/s//tcrv_rvv.low_precision_resource.realization_decision", value = "artifact-name-derived-resource-decision"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REALIZATION-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"/s//tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-performance-win.v1"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-FEEDBACK
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.remediation_statement_strategy", value = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"/s//tcrv_rvv.low_precision_resource.remediation_statement_strategy", value = "metadata-only-packed-i4-unpack-plan"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REMEDIATION-STATEMENT-STRATEGY
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.remediation_product_plan", value = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1"/s//tcrv_rvv.low_precision_resource.remediation_product_plan", value = "metadata-only-packed-i4-product-plan"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REMEDIATION-PRODUCT-PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"/s//tcrv_rvv.low_precision_resource.schedule_decision", value = "metadata-only-packed-i4-schedule-decision"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-SCHEDULE-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.realization_admission_schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"/s//tcrv_rvv.low_precision_resource.realization_admission_schedule_decision", value = "metadata-only-packed-i4-admission-schedule-decision"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-ADMISSION-SCHEDULE-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "false"/s//tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "true"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-SELECTION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_maturity_outcome", value = "no-win"/s//tcrv_rvv.low_precision_resource.performance_maturity_outcome", value = "win"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-OUTCOME
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_admission_closure", value = "no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1"/s//tcrv_rvv.low_precision_resource.performance_admission_closure", value = "metadata-only-no-safe-repair"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-ADMISSION-CLOSURE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "packed-i4-no-provider-beyond-local-repair-facts"/s//tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "metadata-only-beyond-local-blocker"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-BEYOND-LOCAL-BLOCKER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"/s//tcrv_rvv.low_precision_resource.dispatch_preference", value = "performance-preferred"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-PREFERENCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize/s//selected_dispatch_case_mirror:@metadata_only_dispatch_case/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-CASE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback/s//selected_dispatch_fallback_mirror:@metadata_only_scalar_fallback/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-FALLBACK
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"}/s//tcrv_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"}, {key = "tcrv_rvv.low_precision_resource.performance_win_claim_allowed", value = "true"}/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=METADATA-ONLY-WIN-CLAIM
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
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 5 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_unroll_factor = 1 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_feedback = "same-target-packed-i4-no-win.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_baseline = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_best_speedup_range = "0.896848..1.020953"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_action = "no-win-repair-required-before-performance-claim"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_decision = "deny-performance-preferred-with-resource-cost-no-win-blocker"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_closure = "no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_reopen_requirement = "provider-repair-beyond-local-statement-frontier-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_contract = "rvv-low-precision-packed-i4-beyond-local-repair-admission.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_decision = "deny-performance-preferred-no-provider-repair-beyond-local-frontier"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker = "packed-i4-no-provider-beyond-local-repair-facts"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement = "new-typed-provider-beyond-local-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity = "executable-not-performance-mature"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity_evidence = "same-target-packed-i4-local-repair-frontier-no-win-gate4.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_maturity_outcome = "no-win"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_selection_eligible = "false"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.dispatch_preference = "not-performance-preferred"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_plan_contract = "rvv-low-precision-packed-i4-resource-remediation-plan.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_plan = "close-packed-i4-local-statement-repair-frontier-before-performance-claim.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_statement_strategy = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_vector_budget = "packed-i4-remediation-budget-5of32-vector-groups"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_schedule_contract = "rvv-low-precision-packed-i4-resource-remediation-schedule.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_unpack_plan = "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_reduction_plan = "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_vl_plan = "two-region-runtime-avl-product-reduce-then-dequant-store.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.schedule_decision_contract = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.schedule_decision = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.schedule_decision_reason = "accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32"
// REALIZED: tcrv_rvv.vsetvl_region_marker
// REALIZED-SAME: phase = "load-product-reduce"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 1 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"
// REALIZED: tcrv_rvv.gearbox_cross_region_handoff
// REALIZED-SAME: beyond_local_repair_admission_blocker = "packed-i4-no-provider-beyond-local-repair-facts"
// REALIZED-SAME: beyond_local_repair_admission_contract = "rvv-low-precision-packed-i4-beyond-local-repair-admission.v1"
// REALIZED-SAME: beyond_local_repair_admission_decision = "deny-performance-preferred-no-provider-repair-beyond-local-frontier"
// REALIZED-SAME: beyond_local_repair_admission_reopen_requirement = "new-typed-provider-beyond-local-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-SAME: dequant_region_index = 2 : i64
// REALIZED-SAME: from_phase = "load-product-reduce"
// REALIZED-SAME: operand_form = "packed-i4-nibbles"
// REALIZED-SAME: packing_layout = "two-signed-i4-elements-per-byte-low-high-nibbles"
// REALIZED-SAME: peak_live_vector_groups = 5 : i64
// REALIZED-SAME: performance_admission_closure = "no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1"
// REALIZED-SAME: performance_admission_reopen_requirement = "provider-repair-beyond-local-statement-frontier-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-SAME: primitive_chain_contract = "rvv-low-precision-widening-reduction-primitive-facts.v1"
// REALIZED-SAME: primitive_product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"
// REALIZED-SAME: primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"
// REALIZED-SAME: primitive_source_signedness = "signed"
// REALIZED-SAME: product_region_index = 1 : i64
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: remediation_plan = "close-packed-i4-local-statement-repair-frontier-before-performance-claim.v1"
// REALIZED-SAME: remediation_plan_contract = "rvv-low-precision-packed-i4-resource-remediation-plan.v1"
// REALIZED-SAME: remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1"
// REALIZED-SAME: remediation_reduction_plan = "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed.v1"
// REALIZED-SAME: remediation_schedule_contract = "rvv-low-precision-packed-i4-resource-remediation-schedule.v1"
// REALIZED-SAME: remediation_statement_strategy = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"
// REALIZED-SAME: remediation_unpack_plan = "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1"
// REALIZED-SAME: remediation_vector_budget = "packed-i4-remediation-budget-5of32-vector-groups"
// REALIZED-SAME: remediation_vl_plan = "two-region-runtime-avl-product-reduce-then-dequant-store.v1"
// REALIZED-SAME: resource_candidate_set = "rvv-low-precision-direct-contraction-resource-candidate-set.v4[i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,u2-grouped-tail-safe,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1:u1-unpack-required]"
// REALIZED-SAME: resource_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"
// REALIZED-SAME: resource_selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"
// REALIZED-SAME: schedule_decision = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"
// REALIZED-SAME: schedule_decision_contract = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"
// REALIZED-SAME: schedule_decision_reason = "accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32"
// REALIZED-SAME: unpack_intent = "sign-extend-i4-nibbles-before-widening-product"
// REALIZED-SAME: vector_register_budget = 32 : i64
// REALIZED: tcrv_rvv.vsetvl_region_marker
// REALIZED-SAME: phase = "dequant-store"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 2 : i64

// PLAN: {key = "tcrv_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
// PLAN: {key = "tcrv_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_contract", value = "rvv-low-precision-selected-body-realization-admission.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_decision", value = "realize"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_evidence", value = "gate4-packed-i4-high-nibble-vwmacc-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_dispatch_policy", value = "correctness-fallback"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_schedule_decision_contract", value = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realization_admission_schedule_decision_reason", value = "accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.realized_vsetvl_region_count", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_region_index", value = "1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_region_index", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_phase", value = "load-product-reduce"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_phase", value = "dequant-store"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_best_speedup_range", value = "0.896848..1.020953"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_action", value = "no-win-repair-required-before-performance-claim"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_plan_contract", value = "rvv-low-precision-packed-i4-resource-remediation-plan.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_plan", value = "close-packed-i4-local-statement-repair-frontier-before-performance-claim.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_statement_strategy", value = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_vector_budget", value = "packed-i4-remediation-budget-5of32-vector-groups"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_schedule_contract", value = "rvv-low-precision-packed-i4-resource-remediation-schedule.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_unpack_plan", value = "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_product_plan", value = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_reduction_plan", value = "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_vl_plan", value = "two-region-runtime-avl-product-reduce-then-dequant-store.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.schedule_decision_contract", value = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.schedule_decision_reason", value = "accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_decision", value = "deny-performance-preferred-with-resource-cost-no-win-blocker"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_closure", value = "no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_reopen_requirement", value = "provider-repair-beyond-local-statement-frontier-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_contract", value = "rvv-low-precision-packed-i4-beyond-local-repair-admission.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_decision", value = "deny-performance-preferred-no-provider-repair-beyond-local-frontier"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "packed-i4-no-provider-beyond-local-repair-facts"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement", value = "new-typed-provider-beyond-local-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity", value = "executable-not-performance-mature"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity_evidence", value = "same-target-packed-i4-local-repair-frontier-no-win-gate4.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_maturity_outcome", value = "no-win"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_selection_eligible", value = "false"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"}

// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]
// HEADER: tianchenrv.rvv.low_precision_resource.operand_form: packed-i4-nibbles
// HEADER: tianchenrv.rvv.low_precision_resource.source_signedness: signed
// HEADER: tianchenrv.rvv.low_precision_resource.storage_element_width: 8
// HEADER: tianchenrv.rvv.low_precision_resource.effective_element_width: 4
// HEADER: tianchenrv.rvv.low_precision_resource.packing_layout: two-signed-i4-elements-per-byte-low-high-nibbles
// HEADER: tianchenrv.rvv.low_precision_resource.unpack_intent: sign-extend-i4-nibbles-before-widening-product
// HEADER: tianchenrv.rvv.low_precision_resource.peak_live_vector_groups: 5
// HEADER: tianchenrv.rvv.low_precision_resource.realization_decision: consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_contract: rvv-low-precision-selected-body-realization-admission.v1
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_decision: realize
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_evidence: gate4-packed-i4-high-nibble-vwmacc-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_dispatch_policy: correctness-fallback
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision: select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.realization_admission_schedule_decision_reason: accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32
// HEADER: tianchenrv.rvv.low_precision_resource.realized_vsetvl_region_count: 2
// HEADER: tianchenrv.rvv.low_precision_resource.product_region_index: 1
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_region_index: 2
// HEADER: tianchenrv.rvv.low_precision_resource.product_phase: load-product-reduce
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_phase: dequant-store
// HEADER: tianchenrv.rvv.low_precision_resource.performance_feedback: same-target-packed-i4-no-win.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_baseline: scalar-c-reference/product-reduction-dequant-packed-i4-v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_best_speedup_range: 0.896848..1.020953
// HEADER: tianchenrv.rvv.low_precision_resource.performance_action: no-win-repair-required-before-performance-claim
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_plan_contract: rvv-low-precision-packed-i4-resource-remediation-plan.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_plan: close-packed-i4-local-statement-repair-frontier-before-performance-claim.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_statement_strategy: low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_vector_budget: packed-i4-remediation-budget-5of32-vector-groups
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_schedule_contract: rvv-low-precision-packed-i4-resource-remediation-schedule.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_unpack_plan: shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_product_plan: low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_reduction_plan: single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed.v1
// HEADER: tianchenrv.rvv.low_precision_resource.remediation_vl_plan: two-region-runtime-avl-product-reduce-then-dequant-store.v1
// HEADER: tianchenrv.rvv.low_precision_resource.schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1
// HEADER: tianchenrv.rvv.low_precision_resource.schedule_decision: select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.schedule_decision_reason: accepted-remediation-schedule-high-nibble-vwmacc-single-vwredsum-budget-5of32
// HEADER: tianchenrv.rvv.low_precision_resource.performance_admission_decision: deny-performance-preferred-with-resource-cost-no-win-blocker
// HEADER: tianchenrv.rvv.low_precision_resource.performance_admission_closure: no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_admission_reopen_requirement: provider-repair-beyond-local-statement-frontier-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER: tianchenrv.rvv.low_precision_resource.beyond_local_repair_admission_contract: rvv-low-precision-packed-i4-beyond-local-repair-admission.v1
// HEADER: tianchenrv.rvv.low_precision_resource.beyond_local_repair_admission_decision: deny-performance-preferred-no-provider-repair-beyond-local-frontier
// HEADER: tianchenrv.rvv.low_precision_resource.beyond_local_repair_admission_blocker: packed-i4-no-provider-beyond-local-repair-facts
// HEADER: tianchenrv.rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement: new-typed-provider-beyond-local-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity: executable-not-performance-mature
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity_evidence: same-target-packed-i4-local-repair-frontier-no-win-gate4.v1
// HEADER: tianchenrv.rvv.low_precision_resource.performance_maturity_outcome: no-win
// HEADER: tianchenrv.rvv.low_precision_resource.performance_selection_eligible: false
// HEADER: tianchenrv.rvv.low_precision_resource.dispatch_preference: not-performance-preferred
// HEADER: tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case
// HEADER: tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vwmul_vv_i16mf2
// CPP: __riscv_vsra_vx_i16mf2
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vwmacc_vv_i16mf2
// CPP: __riscv_vwredsum_vs_i16mf2_i32m1
// CPP: tcrv_emitc.assign target=dot_acc_vec

// STALE-PACKED-DECISION: requires resource_decision to match the selected low-precision resource candidate
// STALE-PACKED-DECISION: consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1

// STALE-PACKED-REGION-COUNT: requires region_count to match the bounded Gearbox resource decision

// STALE-PACKED-FROM-PHASE: requires from_phase 'load-product-reduce'

// STALE-PACKED-HANDOFF-REMEDIATION: requires packed-i4 remediation attribute 'remediation_statement_strategy' to match provider-owned resource fact 'low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum'
// STALE-PACKED-HANDOFF-REMEDIATION-SAME: metadata-only-packed-i4-unpack-plan

// STALE-PACKED-HANDOFF-REMEDIATION-PRODUCT-PLAN: requires packed-i4 remediation attribute 'remediation_product_plan' to match provider-owned resource fact 'low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1'
// STALE-PACKED-HANDOFF-REMEDIATION-PRODUCT-PLAN-SAME: metadata-only-packed-i4-product-plan

// STALE-PACKED-HANDOFF-SCHEDULE-DECISION: requires packed-i4 remediation attribute 'schedule_decision' to match provider-owned resource fact 'select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1'
// STALE-PACKED-HANDOFF-SCHEDULE-DECISION-SAME: metadata-only-packed-i4-schedule-decision

// MISSING-PACKED-HANDOFF-REMEDIATION: requires packed-i4 remediation attribute 'remediation_vector_budget' on the selected resource handoff

// STALE-ARTIFACT-REALIZATION-DECISION: low_precision_resource.realization_decision provenance must mirror provider-selected low-precision direct-contraction resource realization decision 'consume-low-precision-packed-i4-high-nibble-vwmacc-single-reduce-budget-5of32.v1'
// STALE-ARTIFACT-REALIZATION-DECISION-SAME: artifact-name-derived-resource-decision

// STALE-ARTIFACT-PERFORMANCE-FEEDBACK: low_precision_resource.performance_feedback provenance must mirror provider-selected low-precision direct-contraction resource performance feedback 'same-target-packed-i4-no-win.v1'
// STALE-ARTIFACT-PERFORMANCE-FEEDBACK-SAME: same-target-packed-i4-performance-win.v1

// STALE-ARTIFACT-REMEDIATION-STATEMENT-STRATEGY: low_precision_resource.remediation_statement_strategy provenance must mirror provider-selected low-precision direct-contraction resource remediation statement strategy 'low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum'
// STALE-ARTIFACT-REMEDIATION-STATEMENT-STRATEGY-SAME: metadata-only-packed-i4-unpack-plan

// STALE-ARTIFACT-REMEDIATION-PRODUCT-PLAN: low_precision_resource.remediation_product_plan provenance must mirror provider-selected low-precision direct-contraction resource remediation product plan 'low-shifted-product-i16-rescale-plus-high-nibble-vwmacc.v1'
// STALE-ARTIFACT-REMEDIATION-PRODUCT-PLAN-SAME: metadata-only-packed-i4-product-plan

// STALE-ARTIFACT-SCHEDULE-DECISION: low_precision_resource.schedule_decision provenance must mirror provider-selected low-precision direct-contraction resource schedule decision 'select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1'
// STALE-ARTIFACT-SCHEDULE-DECISION-SAME: metadata-only-packed-i4-schedule-decision
// STALE-ARTIFACT-ADMISSION-SCHEDULE-DECISION: low_precision_resource.realization_admission_schedule_decision provenance must mirror provider-selected low-precision direct-contraction resource realization admission schedule decision 'select-packed-i4-high-nibble-vwmacc-single-reduce-u1-two-region-budget-5of32.v1'
// STALE-ARTIFACT-ADMISSION-SCHEDULE-DECISION-SAME: metadata-only-packed-i4-admission-schedule-decision

// STALE-ARTIFACT-PERFORMANCE-SELECTION: low_precision_resource.performance_selection_eligible provenance must mirror provider-selected low-precision direct-contraction resource performance selection eligibility 'false'
// STALE-ARTIFACT-PERFORMANCE-SELECTION-SAME: true

// STALE-ARTIFACT-PERFORMANCE-OUTCOME: low_precision_resource.performance_maturity_outcome provenance must mirror provider-selected low-precision direct-contraction resource performance maturity outcome 'no-win'
// STALE-ARTIFACT-PERFORMANCE-OUTCOME-SAME: win

// STALE-ARTIFACT-PERFORMANCE-ADMISSION-CLOSURE: low_precision_resource.performance_admission_closure provenance must mirror provider-selected low-precision direct-contraction resource performance admission closure 'no-safe-local-packed-i4-repair-frontier-loop-11-budget-5of32.v1'
// STALE-ARTIFACT-PERFORMANCE-ADMISSION-CLOSURE-SAME: metadata-only-no-safe-repair

// STALE-ARTIFACT-BEYOND-LOCAL-BLOCKER: low_precision_resource.beyond_local_repair_admission_blocker provenance must mirror provider-selected low-precision direct-contraction resource beyond-local repair admission blocker 'packed-i4-no-provider-beyond-local-repair-facts'
// STALE-ARTIFACT-BEYOND-LOCAL-BLOCKER-SAME: metadata-only-beyond-local-blocker

// STALE-ARTIFACT-DISPATCH-PREFERENCE: low_precision_resource.dispatch_preference provenance must mirror provider-selected low-precision direct-contraction resource dispatch preference 'not-performance-preferred'
// STALE-ARTIFACT-DISPATCH-PREFERENCE-SAME: performance-preferred

// STALE-ARTIFACT-DISPATCH-CASE: selected_dispatch_case_mirror
// STALE-ARTIFACT-DISPATCH-CASE-SAME: provider-owned selected-dispatch low-precision policy boundary case mirror
// STALE-ARTIFACT-DISPATCH-CASE-SAME: metadata_only_dispatch_case

// STALE-ARTIFACT-DISPATCH-FALLBACK: selected_dispatch_fallback_mirror
// STALE-ARTIFACT-DISPATCH-FALLBACK-SAME: provider-owned selected-dispatch low-precision policy boundary fallback mirror
// STALE-ARTIFACT-DISPATCH-FALLBACK-SAME: metadata_only_scalar_fallback

// METADATA-ONLY-WIN-CLAIM: rejects metadata-only packed-i4 performance win-claim allowance 'true'
// METADATA-ONLY-WIN-CLAIM-SAME: performance_win_claim_allowed
// METADATA-ONLY-WIN-CLAIM-SAME: no-win-repair-required-before-performance-claim
