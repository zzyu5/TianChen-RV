// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// Stage 3 single-scope flip: the resource decision now lives on the lone with_vl
// scope as weft_rvv.low_precision_resource.realization_decision (the deleted
// two-scope handoff op previously carried a bare resource_decision). Re-targeted to
// the with_vl occurrence; still fail-closed at emission-plan time with the specific
// "requires packed-i4 realization decision" provider-fact-gate reason.
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.realization_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1"/weft_rvv.low_precision_resource.realization_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-DECISION
// Stage 3 single-scope flip: the region count now lives on with_vl as
// weft_rvv.low_precision_resource.realized_vsetvl_region_count (the deleted handoff/
// markers previously carried region_count). Re-targeted; still fail-closed at
// emission-plan time with the specific "realized vsetvl region count" reason.
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64/weft_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-REGION-COUNT
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.product_phase = "load-product-reduce"/weft_rvv.low_precision_resource.product_phase = "tail-product-reduce"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-FROM-PHASE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.remediation_statement_strategy = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue"/weft_rvv.low_precision_resource.remediation_statement_strategy = "metadata-only-packed-i4-unpack-plan"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-REMEDIATION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1"/weft_rvv.low_precision_resource.remediation_product_plan = "metadata-only-packed-i4-product-plan"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-REMEDIATION-PRODUCT-PLAN
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.packed_unpack_plan = "low-high-i4-sign-extend-to-i8mf4"/weft_rvv.low_precision_resource.packed_unpack_plan = "metadata-only-packed-i4-unpack-plan"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-UNPACK-PLAN
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.schedule_decision = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"/weft_rvv.low_precision_resource.schedule_decision = "metadata-only-packed-i4-schedule-decision"/' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-HANDOFF-SCHEDULE-DECISION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed 's/weft_rvv.low_precision_resource.remediation_vector_budget = "packed-i4-remediation-budget-5of32-vector-groups", //' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-PACKED-HANDOFF-REMEDIATION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries | sed '0,/weft_rvv.low_precision_resource.performance_feedback = "same-target-packed-i4-no-win.v1", /s///' | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-REALIZATION-POLICY-EVIDENCE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1"/s//weft_rvv.low_precision_resource.realization_decision", value = "artifact-name-derived-resource-decision"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REALIZATION-DECISION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"/s//weft_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-performance-win.v1"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-FEEDBACK
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.remediation_statement_strategy", value = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue"/s//weft_rvv.low_precision_resource.remediation_statement_strategy", value = "metadata-only-packed-i4-unpack-plan"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REMEDIATION-STATEMENT-STRATEGY
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.remediation_product_plan", value = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1"/s//weft_rvv.low_precision_resource.remediation_product_plan", value = "metadata-only-packed-i4-product-plan"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-REMEDIATION-PRODUCT-PLAN
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.packed_unpack_plan", value = "low-high-i4-sign-extend-to-i8mf4"/s//weft_rvv.low_precision_resource.packed_unpack_plan", value = "metadata-only-packed-i4-unpack-plan"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PACKED-UNPACK-PLAN
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"/s//weft_rvv.low_precision_resource.schedule_decision", value = "metadata-only-packed-i4-schedule-decision"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-SCHEDULE-DECISION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.realization_admission_schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"/s//weft_rvv.low_precision_resource.realization_admission_schedule_decision", value = "metadata-only-packed-i4-admission-schedule-decision"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-ADMISSION-SCHEDULE-DECISION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.performance_selection_eligible", value = "false"/s//weft_rvv.low_precision_resource.performance_selection_eligible", value = "true"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-SELECTION
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.performance_maturity_outcome", value = "no-win"/s//weft_rvv.low_precision_resource.performance_maturity_outcome", value = "win"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-OUTCOME
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.performance_admission_closure", value = "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1"/s//weft_rvv.low_precision_resource.performance_admission_closure", value = "metadata-only-no-safe-repair"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-PERFORMANCE-ADMISSION-CLOSURE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win"/s//weft_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "metadata-only-beyond-local-blocker"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-BEYOND-LOCAL-BLOCKER
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"/s//weft_rvv.low_precision_resource.dispatch_preference", value = "performance-preferred"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-PREFERENCE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize/s//selected_dispatch_case_mirror:@metadata_only_dispatch_case/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-CASE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback/s//selected_dispatch_fallback_mirror:@metadata_only_scalar_fallback/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-DISPATCH-FALLBACK
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.selected_dispatch_preference", value = "not-performance-preferred"/s//weft_rvv.low_precision_resource.selected_dispatch_preference", value = "performance-preferred"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY-DISPATCH-PREFERENCE
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.performance_win_claim_allowed", value = "false"/s//weft_rvv.low_precision_resource.performance_win_claim_allowed", value = "true"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY-WIN-CLAIM
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-rvv-emitc-to-cpp | FileCheck %s --check-prefix=CPP

// Focused measurement-disposition fixture for the same product-reduction-dequant op kind with
// an explicit signed packed-i4 selected resource. The candidate is authority
// only because it is carried in the typed pre-realized weft_rvv body and then
// consumed by RVV selected-body realization/provider planning.

module {
  weft.exec.kernel @pre_realized_body_product_reduce_dequantize_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", weft_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// REALIZED-DAG: weft_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"
// REALIZED-DAG: weft_rvv.low_precision_resource.candidate_count = 3 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.legal_candidate_count = 3 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.selected_candidate_index = 3 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.operand_form = "packed-i4-nibbles"
// REALIZED-DAG: weft_rvv.low_precision_resource.storage_element_width = 8 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.effective_element_width = 4 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.packing_layout = "two-signed-i4-elements-per-byte-low-high-nibbles"
// REALIZED-DAG: weft_rvv.low_precision_resource.unpack_intent = "sign-extend-i4-nibbles-before-widening-product"
// REALIZED-DAG: weft_rvv.low_precision_resource.packed_load_unpack_contract = "rvv-packed-i4-load-unpack-resource-facts.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.packed_storage_load = "unit-stride-vle8-i8mf4-packed-i4x2"
// REALIZED-DAG: weft_rvv.low_precision_resource.packed_unpack_plan = "low-high-i4-sign-extend-to-i8mf4"
// REALIZED-DAG: weft_rvv.low_precision_resource.packed_unpacked_source = "signed-i8mf4-logical-lanes-from-packed-i4x2"
// REALIZED-DAG: weft_rvv.low_precision_resource.realization_decision = "consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.realized_peak_live_vector_groups = 5 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.realized_unroll_factor = 1 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_feedback = "same-target-packed-i4-no-win.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_baseline = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_best_speedup_range = "0.895307..1.027027"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_action = "no-win-repair-required-before-performance-claim"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_admission_decision = "deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_admission_closure = "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_admission_reopen_requirement = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.beyond_local_repair_admission_contract = "rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.beyond_local_repair_admission_decision = "deny-performance-preferred-campaign-no-further-provider-repair"
// REALIZED-DAG: weft_rvv.low_precision_resource.beyond_local_repair_admission_blocker = "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win"
// REALIZED-DAG: weft_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_maturity = "executable-not-performance-mature"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_maturity_evidence = "same-target-packed-i4-campaign-no-further-repair-no-win-gate4.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_maturity_outcome = "no-win"
// REALIZED-DAG: weft_rvv.low_precision_resource.performance_selection_eligible = "false"
// REALIZED-DAG: weft_rvv.low_precision_resource.dispatch_preference = "not-performance-preferred"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_plan_contract = "rvv-low-precision-packed-i4-resource-remediation-plan.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_plan = "attempt-packed-i4-beyond-local-scalar-epilogue-before-performance-claim.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_statement_strategy = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_vector_budget = "packed-i4-remediation-budget-5of32-vector-groups"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_schedule_contract = "rvv-low-precision-packed-i4-resource-remediation-schedule.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_unpack_plan = "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_reduction_plan = "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed-scalar-epilogue.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.remediation_vl_plan = "two-region-runtime-avl-product-reduce-then-scalar-epilogue-store.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.schedule_decision_contract = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.schedule_decision = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"
// REALIZED-DAG: weft_rvv.low_precision_resource.schedule_decision_reason = "accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32"
// Stage 3 single-scope packed-i4 flip: the realized body is one weft_rvv.with_vl
// scope carrying a typed weft_rvv.packed_i4_nibble_unpack_product head + the inline
// dequant chain, with NO weft_rvv.gearbox_cross_region_handoff carrier and NO
// weft_rvv.vsetvl_region_marker placeholders. The structural unroll_factor (=1) the
// conversion reads is stamped on with_vl; the low_precision_resource.* facts above
// survive on the single scope. Numerics HW-validated on ssh rvv (tolerance=1e-05).
// REALIZED-DAG: unroll_factor = 1 : i64
// REALIZED-DAG: weft_rvv.packed_i4_nibble_unpack_product
// REALIZED-DAG: kind = "signed_packed_i4_nibble_unpack_product"
// REALIZED-DAG: weft_rvv.standalone_reduce
// REALIZED-DAG: weft_rvv.dequantize
// REALIZED-DAG: weft_rvv.store
// The deleted two-scope carrier/markers are genuinely absent across the WHOLE
// realized body (the REALIZED RUN line's --implicit-check-not enforces this
// globally, fail-closed: a stray handoff or marker anywhere = incomplete flip).

// PLAN: {key = "weft_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
// PLAN: {key = "weft_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.selected_dispatch_policy_contract", value = "rvv-low-precision-packed-i4-dispatch-performance-policy.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.dispatch_policy_path", value = "correctness-fallback"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.selected_dispatch_preference", value = "not-performance-preferred"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_preference_denial_reason", value = "same-target-measurement-no-win-or-regression"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.fallback_reason", value = "same-target-measurement-no-win-or-regression"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.route_support_allowed", value = "true"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.correctness_execution_allowed", value = "true"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_selection_allowed", value = "false"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_win_claim_allowed", value = "false"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.correctness_fallback_path_selected", value = "true"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_preferred_path_selected", value = "false"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.resource_owner_mirror_source", value = "provider-owned-low-precision-contraction-resource-selection.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.candidate_count", value = "3"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.legal_candidate_count", value = "3"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.selected_candidate_index", value = "3"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.packed_load_unpack_contract", value = "rvv-packed-i4-load-unpack-resource-facts.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.packed_storage_load", value = "unit-stride-vle8-i8mf4-packed-i4x2"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.packed_unpack_plan", value = "low-high-i4-sign-extend-to-i8mf4"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.packed_unpacked_source", value = "signed-i8mf4-logical-lanes-from-packed-i4x2"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_decision", value = "consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realized_vsetvl_region_count", value = "2"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.product_region_index", value = "1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.dequant_region_index", value = "2"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.product_phase", value = "load-product-reduce"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.dequant_phase", value = "dequant-store"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.resource_cost_contract", value = "rvv-low-precision-packed-i4-resource-cost-contract.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.resource_cost_blocker", value = "packed-i4-loop-11-budget-5of32-resource-cost-boundary"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.schedule_decision_contract", value = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.schedule_decision_reason", value = "accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_contract", value = "rvv-low-precision-selected-body-realization-admission.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_decision", value = "realize"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_evidence", value = "gate4-packed-i4-scalar-epilogue-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_dispatch_policy", value = "correctness-fallback"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_schedule_decision_contract", value = "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_schedule_decision", value = "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.realization_admission_schedule_decision_reason", value = "accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_feedback", value = "same-target-packed-i4-no-win.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference/product-reduction-dequant-packed-i4-v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_best_speedup_range", value = "0.895307..1.027027"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_action", value = "no-win-repair-required-before-performance-claim"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_plan_contract", value = "rvv-low-precision-packed-i4-resource-remediation-plan.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_plan", value = "attempt-packed-i4-beyond-local-scalar-epilogue-before-performance-claim.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_statement_strategy", value = "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_vector_budget", value = "packed-i4-remediation-budget-5of32-vector-groups"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_schedule_contract", value = "rvv-low-precision-packed-i4-resource-remediation-schedule.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_unpack_plan", value = "shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_product_plan", value = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_reduction_plan", value = "single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed-scalar-epilogue.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.remediation_vl_plan", value = "two-region-runtime-avl-product-reduce-then-scalar-epilogue-store.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_admission_decision", value = "deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_admission_closure", value = "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_admission_reopen_requirement", value = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.beyond_local_repair_admission_contract", value = "rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.beyond_local_repair_admission_decision", value = "deny-performance-preferred-campaign-no-further-provider-repair"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement", value = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_maturity", value = "executable-not-performance-mature"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_maturity_evidence", value = "same-target-packed-i4-campaign-no-further-repair-no-win-gate4.v1"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_maturity_outcome", value = "no-win"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.performance_selection_eligible", value = "false"}
// PLAN-DAG: {key = "weft_rvv.low_precision_resource.dispatch_preference", value = "not-performance-preferred"}

// HEADER: weft.rvv.low_precision_resource.resource_owner_mirror.source: provider-owned-low-precision-contraction-resource-selection.v1
// HEADER: weft.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]
// HEADER: weft.rvv.low_precision_resource.candidate_count: 3
// HEADER: weft.rvv.low_precision_resource.legal_candidate_count: 3
// HEADER: weft.rvv.low_precision_resource.selected_candidate_index: 3
// HEADER: weft.rvv.low_precision_resource.operand_form: packed-i4-nibbles
// HEADER: weft.rvv.low_precision_resource.source_signedness: signed
// HEADER: weft.rvv.low_precision_resource.storage_element_width: 8
// HEADER: weft.rvv.low_precision_resource.effective_element_width: 4
// HEADER: weft.rvv.low_precision_resource.packing_layout: two-signed-i4-elements-per-byte-low-high-nibbles
// HEADER: weft.rvv.low_precision_resource.unpack_intent: sign-extend-i4-nibbles-before-widening-product
// HEADER: weft.rvv.low_precision_resource.packed_load_unpack_contract: rvv-packed-i4-load-unpack-resource-facts.v1
// HEADER: weft.rvv.low_precision_resource.packed_storage_load: unit-stride-vle8-i8mf4-packed-i4x2
// HEADER: weft.rvv.low_precision_resource.packed_unpack_plan: low-high-i4-sign-extend-to-i8mf4
// HEADER: weft.rvv.low_precision_resource.packed_unpacked_source: signed-i8mf4-logical-lanes-from-packed-i4x2
// HEADER: weft.rvv.low_precision_resource.peak_live_vector_groups: 5
// HEADER: weft.rvv.low_precision_resource.realization_decision: consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_contract: rvv-low-precision-selected-body-realization-admission.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_decision: realize
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_evidence: gate4-packed-i4-scalar-epilogue-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_dispatch_policy: correctness-fallback
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_schedule_decision: select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.realization_admission_schedule_decision_reason: accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32
// HEADER: weft.rvv.low_precision_resource.realized_vsetvl_region_count: 2
// HEADER: weft.rvv.low_precision_resource.product_region_index: 1
// HEADER: weft.rvv.low_precision_resource.dequant_region_index: 2
// HEADER: weft.rvv.low_precision_resource.product_phase: load-product-reduce
// HEADER: weft.rvv.low_precision_resource.dequant_phase: dequant-store
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_feedback: same-target-packed-i4-no-win.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_baseline: scalar-c-reference/product-reduction-dequant-packed-i4-v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_best_speedup_range: 0.895307..1.027027
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_action: no-win-repair-required-before-performance-claim
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_plan_contract: rvv-low-precision-packed-i4-resource-remediation-plan.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_plan: attempt-packed-i4-beyond-local-scalar-epilogue-before-performance-claim.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_statement_strategy: low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_vector_budget: packed-i4-remediation-budget-5of32-vector-groups
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_schedule_contract: rvv-low-precision-packed-i4-resource-remediation-schedule.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_unpack_plan: shift-left-low-signed-i4-nibbles-and-shift-right-high-nibbles.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_product_plan: low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_reduction_plan: single-vwredsum-i16-high-vwmacc-pair-sum-with-i32-seed-scalar-epilogue.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_vl_plan: two-region-runtime-avl-product-reduce-then-scalar-epilogue-store.v1
// HEADER: weft.rvv.low_precision_resource.schedule_decision_contract: rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1
// HEADER: weft.rvv.low_precision_resource.schedule_decision: select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1
// HEADER: weft.rvv.low_precision_resource.schedule_decision_reason: accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_decision: deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_closure: no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_reopen_requirement: new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_contract: rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_decision: deny-performance-preferred-campaign-no-further-provider-repair
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_blocker: packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_reopen_requirement: new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_maturity: executable-not-performance-mature
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_maturity_evidence: same-target-packed-i4-campaign-no-further-repair-no-win-gate4.v1
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_maturity_outcome: no-win
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_selection_eligible: false
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.selected_dispatch_policy_contract: rvv-low-precision-packed-i4-dispatch-performance-policy.v1
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.dispatch_policy_path: correctness-fallback
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.selected_dispatch_preference: not-performance-preferred
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_preference_denial_reason: same-target-measurement-no-win-or-regression
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.fallback_reason: same-target-measurement-no-win-or-regression
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.route_support_allowed: true
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.correctness_execution_allowed: true
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_selection_allowed: false
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_win_claim_allowed: false
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.correctness_fallback_path_selected: true
// HEADER-DAG: weft.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_preferred_path_selected: false
// HEADER: weft.rvv.low_precision_resource.measurement_disposition_evidence_mirror.dispatch_preference: not-performance-preferred
// HEADER: weft.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case
// HEADER: weft.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope
// HEADER: void weft_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

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
// CPP: weft_emitc.assign target=dot_acc_vec
// CPP: __riscv_vmv_x_s_i32m1_i32
// Stage 3 single-scope flip: the conversion emits the UNIFIED vector dequant
// epilogue (lane-0 extract -> scalar dequant -> VL=1 f32 splat-store), the same as
// the grouped/unpacked candidate -- numerically a single-scalar write to out[0]
// (HW-validated on ssh rvv, tolerance=1e-05). The legacy packed-i4 scalar out[0]=
// store is retired (it was the other code path's form, numerically equivalent).
// CPP: __riscv_vfmv_v_f_f32m1
// CPP: __riscv_vse32_v_f32m1

// STALE-PACKED-DECISION: requires packed-i4 realization decision 'consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1' but found 'consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1'

// STALE-PACKED-REGION-COUNT: realized vsetvl region count 2 but found 3

// STALE-PACKED-FROM-PHASE: product phase 'load-product-reduce' but found 'tail-product-reduce'

// STALE-PACKED-HANDOFF-REMEDIATION: measurement-disposition remediation statement strategy 'low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue' but found 'metadata-only-packed-i4-unpack-plan'

// STALE-PACKED-HANDOFF-REMEDIATION-PRODUCT-PLAN: measurement-disposition remediation product plan 'low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1' but found 'metadata-only-packed-i4-product-plan'

// STALE-PACKED-HANDOFF-UNPACK-PLAN: packed-i4 unpack plan 'low-high-i4-sign-extend-to-i8mf4' but found 'metadata-only-packed-i4-unpack-plan'

// STALE-PACKED-HANDOFF-SCHEDULE-DECISION: packed-i4 schedule decision 'select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1' but found 'metadata-only-packed-i4-schedule-decision'

// MISSING-PACKED-HANDOFF-REMEDIATION: requires low-precision measurement-disposition evidence/admission fact 'weft_rvv.low_precision_resource.remediation_vector_budget' before policy/evidence validation

// MISSING-REALIZATION-POLICY-EVIDENCE: requires low-precision measurement-disposition evidence/admission fact 'weft_rvv.low_precision_resource.performance_feedback' before policy/evidence validation

// STALE-ARTIFACT-REALIZATION-DECISION: metadata key '{{.*}}low_precision_resource.realization_decision'{{.*}}'consume-low-precision-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-budget-5of32.v1' but was 'artifact-name-derived-resource-decision'

// STALE-ARTIFACT-PERFORMANCE-FEEDBACK: metadata key '{{.*}}low_precision_resource.performance_feedback'{{.*}}'same-target-packed-i4-no-win.v1' but was 'same-target-packed-i4-performance-win.v1'

// STALE-ARTIFACT-REMEDIATION-STATEMENT-STRATEGY: metadata key '{{.*}}low_precision_resource.remediation_statement_strategy'{{.*}}'low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum-scalar-epilogue' but was 'metadata-only-packed-i4-unpack-plan'

// STALE-ARTIFACT-REMEDIATION-PRODUCT-PLAN: metadata key '{{.*}}low_precision_resource.remediation_product_plan'{{.*}}'low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1' but was 'metadata-only-packed-i4-product-plan'

// STALE-ARTIFACT-PACKED-UNPACK-PLAN: metadata key '{{.*}}low_precision_resource.packed_unpack_plan'{{.*}}'low-high-i4-sign-extend-to-i8mf4' but was 'metadata-only-packed-i4-unpack-plan'

// STALE-ARTIFACT-SCHEDULE-DECISION: metadata key '{{.*}}low_precision_resource.schedule_decision'{{.*}}'select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1' but was 'metadata-only-packed-i4-schedule-decision'
// STALE-ARTIFACT-ADMISSION-SCHEDULE-DECISION: metadata key '{{.*}}low_precision_resource.realization_admission_schedule_decision'{{.*}}'select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1' but was 'metadata-only-packed-i4-admission-schedule-decision'

// STALE-ARTIFACT-PERFORMANCE-SELECTION: metadata key '{{.*}}low_precision_resource.performance_selection_eligible'{{.*}}'false' but was 'true'

// STALE-ARTIFACT-PERFORMANCE-OUTCOME: metadata key '{{.*}}low_precision_resource.performance_maturity_outcome'{{.*}}'no-win' but was 'win'

// STALE-ARTIFACT-PERFORMANCE-ADMISSION-CLOSURE: metadata key '{{.*}}low_precision_resource.performance_admission_closure'{{.*}}'no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1' but was 'metadata-only-no-safe-repair'

// STALE-ARTIFACT-BEYOND-LOCAL-BLOCKER: metadata key '{{.*}}low_precision_resource.beyond_local_repair_admission_blocker'{{.*}}'packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win' but was 'metadata-only-beyond-local-blocker'

// STALE-ARTIFACT-DISPATCH-PREFERENCE: metadata key '{{.*}}low_precision_resource.dispatch_preference'{{.*}}'not-performance-preferred' but was 'performance-preferred'

// STALE-ARTIFACT-DISPATCH-CASE: dispatch case facts 'selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequantize;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case' but was 'selected_dispatch_case_mirror:@metadata_only_dispatch_case;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-case'

// STALE-ARTIFACT-DISPATCH-FALLBACK: dispatch fallback facts 'selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope' but was 'selected_dispatch_fallback_mirror:@metadata_only_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope'

// STALE-POLICY-DISPATCH-PREFERENCE: metadata key '{{.*}}low_precision_resource.selected_dispatch_preference'{{.*}}'not-performance-preferred' but was 'performance-preferred'

// STALE-POLICY-WIN-CLAIM: metadata key '{{.*}}low_precision_resource.performance_win_claim_allowed'{{.*}}'false' but was 'true'
