// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference\/product-reduction-dequant-clamp-packed-i4-v1"/s//tcrv_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference\/product-reduction-dequant-packed-i4-v1"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-BASELINE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.remediation_measurement_evidence", value = "gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh\/widening_product_reduce_dequant_clamp_f32\/same_target_measurement_evidence.json"/s//tcrv_rvv.low_precision_resource.remediation_measurement_evidence", value = "gate4-packed-i4-scalar-epilogue-dequant-ssh\/widening_product_reduce_dequantize_f32\/same_target_measurement_evidence.json"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ARTIFACT-MEASUREMENT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.selected_dispatch_preference", value = "not-performance-preferred"/s//tcrv_rvv.low_precision_resource.selected_dispatch_preference", value = "performance-preferred"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY-DISPATCH-PREFERENCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.correctness_fallback_path_selected", value = "true"/s//tcrv_rvv.low_precision_resource.correctness_fallback_path_selected", value = "false"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-POLICY-CORRECTNESS-FALLBACK
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-rvv-emitc-to-cpp | FileCheck %s --check-prefix=CPP

// Focused dequant-clamp realized-body consumption representative. The selected candidate is
// authority only because it is present on the typed RVV body and consumed by
// the RVV provider/resource planner; artifact names and measurement metadata
// are mirrors that must match the provider facts.

module {
  tcrv.exec.kernel @pre_realized_body_product_reduce_dequant_clamp_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_product_reduce_dequant_clamp attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:lower", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:upper", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body %lhs, %rhs, %acc, %scale, %lower, %upper, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-clamped-dequantized-f32-vector-to-output-buffer", lower_predicate_kind = "slt", memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32", op_kind = "widening_product_reduce_dequant_clamp_f32", tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", source_lmul = "mf4", source_sew = 8 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_product_reduce_dequant_clamp {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope"}
    }
  }
}

// REALIZED-DAG: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.candidate_count = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.legal_candidate_count = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.selected_candidate_index = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.clamp_compare_select_phase = "lower-then-upper-compare-select"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.clamp_phase = "dequant-clamp-store"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.clamp_region_index = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.clamp_select_layout = "clamp-lower-then-upper"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.operand_form = "packed-i4-nibbles"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packed_load_unpack_contract = "rvv-packed-i4-load-unpack-resource-facts.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packed_storage_load = "unit-stride-vle8-i8mf4-packed-i4x2"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packed_unpack_plan = "low-high-i4-sign-extend-to-i8mf4"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packed_unpacked_source = "signed-i8mf4-logical-lanes-from-packed-i4x2"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_baseline = "scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_decision = "deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_closure = "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.performance_admission_reopen_requirement = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_contract = "rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_decision = "deny-performance-preferred-campaign-no-further-provider-repair"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker = "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.remediation_measurement_evidence = "gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json"
// REALIZED: tcrv_rvv.gearbox_cross_region_handoff
// REALIZED-DAG: clamp_compare_select_phase = "lower-then-upper-compare-select"
// REALIZED-DAG: clamp_phase = "dequant-clamp-store"
// REALIZED-DAG: clamp_region_index = 2 : i64
// REALIZED-DAG: clamp_select_layout = "clamp-lower-then-upper"
// REALIZED-DAG: resource_candidate_count = 3 : i64
// REALIZED-DAG: resource_legal_candidate_count = 3 : i64
// REALIZED-DAG: resource_selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"
// REALIZED-DAG: resource_selected_candidate_index = 3 : i64
// REALIZED-DAG: operand_form = "packed-i4-nibbles"
// REALIZED-DAG: packed_load_unpack_contract = "rvv-packed-i4-load-unpack-resource-facts.v1"
// REALIZED-DAG: packed_unpack_plan = "low-high-i4-sign-extend-to-i8mf4"
// REALIZED-DAG: remediation_product_plan = "low-shifted-product-i16-rescale-plus-high-nibble-vwmacc-scalar-epilogue.v1"
// REALIZED: tcrv_rvv.dequantize
// REALIZED: tcrv_rvv.compare
// REALIZED: tcrv_rvv.select

// PLAN: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"}
// PLAN-DAG: {key = "tcrv_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequant_clamp;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case"}
// PLAN-DAG: {key = "tcrv_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.selected_dispatch_policy_contract", value = "rvv-low-precision-packed-i4-dispatch-performance-policy.v1"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.dispatch_policy_path", value = "correctness-fallback"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.selected_dispatch_preference", value = "not-performance-preferred"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.performance_preference_denial_reason", value = "same-target-measurement-no-win-or-regression"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.fallback_reason", value = "same-target-measurement-no-win-or-regression"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.route_support_allowed", value = "true"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.correctness_execution_allowed", value = "true"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.performance_selection_allowed", value = "false"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.performance_win_claim_allowed", value = "false"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.correctness_fallback_path_selected", value = "true"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.performance_preferred_path_selected", value = "false"}
// PLAN-DAG: {key = "tcrv_rvv.low_precision_resource.resource_owner_mirror_source", value = "provider-owned-low-precision-contraction-resource-selection.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.candidate_count", value = "3"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.legal_candidate_count", value = "3"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.selected_candidate_index", value = "3"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.operand_form", value = "packed-i4-nibbles"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.packed_load_unpack_contract", value = "rvv-packed-i4-load-unpack-resource-facts.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.packed_storage_load", value = "unit-stride-vle8-i8mf4-packed-i4x2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.packed_unpack_plan", value = "low-high-i4-sign-extend-to-i8mf4"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.packed_unpacked_source", value = "signed-i8mf4-logical-lanes-from-packed-i4x2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_region_index", value = "1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_region_index", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.product_phase", value = "load-product-reduce"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.dequant_phase", value = "dequant-store"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.clamp_region_index", value = "2"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.clamp_phase", value = "dequant-clamp-store"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.clamp_compare_select_phase", value = "lower-then-upper-compare-select"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.clamp_select_layout", value = "clamp-lower-then-upper"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_baseline", value = "scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.remediation_measurement_evidence", value = "gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_decision", value = "deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_closure", value = "no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.performance_admission_reopen_requirement", value = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_contract", value = "rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_decision", value = "deny-performance-preferred-campaign-no-further-provider-repair"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_blocker", value = "packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win"}
// PLAN: {key = "tcrv_rvv.low_precision_resource.beyond_local_repair_admission_reopen_requirement", value = "new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1"}
// PLAN-SAME: status = "supported"

// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n
// HEADER: tianchenrv.rvv.low_precision_resource.resource_owner_mirror.source: provider-owned-low-precision-contraction-resource-selection.v1
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1,u1-unpack-required]
// HEADER: tianchenrv.rvv.low_precision_resource.candidate_count: 3
// HEADER: tianchenrv.rvv.low_precision_resource.legal_candidate_count: 3
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate_index: 3
// HEADER: tianchenrv.rvv.low_precision_resource.operand_form: packed-i4-nibbles
// HEADER: tianchenrv.rvv.low_precision_resource.packed_load_unpack_contract: rvv-packed-i4-load-unpack-resource-facts.v1
// HEADER: tianchenrv.rvv.low_precision_resource.packed_storage_load: unit-stride-vle8-i8mf4-packed-i4x2
// HEADER: tianchenrv.rvv.low_precision_resource.packed_unpack_plan: low-high-i4-sign-extend-to-i8mf4
// HEADER: tianchenrv.rvv.low_precision_resource.packed_unpacked_source: signed-i8mf4-logical-lanes-from-packed-i4x2
// HEADER: tianchenrv.rvv.low_precision_resource.product_region_index: 1
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_region_index: 2
// HEADER: tianchenrv.rvv.low_precision_resource.product_phase: load-product-reduce
// HEADER: tianchenrv.rvv.low_precision_resource.dequant_phase: dequant-store
// HEADER: tianchenrv.rvv.low_precision_resource.clamp_region_index: 2
// HEADER: tianchenrv.rvv.low_precision_resource.clamp_phase: dequant-clamp-store
// HEADER: tianchenrv.rvv.low_precision_resource.clamp_compare_select_phase: lower-then-upper-compare-select
// HEADER: tianchenrv.rvv.low_precision_resource.clamp_select_layout: clamp-lower-then-upper
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_baseline: scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.remediation_measurement_evidence: gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_decision: deny-performance-preferred-with-campaign-no-further-repair-no-win-blocker
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_closure: no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.performance_admission_reopen_requirement: new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_contract: rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_decision: deny-performance-preferred-campaign-no-further-provider-repair
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_blocker: packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win
// HEADER: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.beyond_local_repair_admission_reopen_requirement: new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.selected_dispatch_policy_contract: rvv-low-precision-packed-i4-dispatch-performance-policy.v1
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.dispatch_policy_path: correctness-fallback
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.selected_dispatch_preference: not-performance-preferred
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_preference_denial_reason: same-target-measurement-no-win-or-regression
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.fallback_reason: same-target-measurement-no-win-or-regression
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.route_support_allowed: true
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.correctness_execution_allowed: true
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_selection_allowed: false
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_win_claim_allowed: false
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.correctness_fallback_path_selected: true
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_dispatch_policy_output_mirror.performance_preferred_path_selected: false
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.measurement_disposition_evidence_mirror.dispatch_preference: not-performance-preferred
// HEADER-DAG: tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequant_clamp;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case
// HEADER-DAG: tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

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
// CPP: __riscv_vmv_x_s_i32m1_i32
// CPP-NOT: __riscv_vfmv_v_f_f32m1
// CPP: __builtin_fmaxf
// CPP-NOT: __riscv_vmflt_vv_f32m1_b32
// CPP: __builtin_fminf
// CPP-NOT: __riscv_vmerge_vvm_f32m1
// CPP: tcrv_emitc.assign target=out[0]
// CPP: [0] =
// CPP-NOT: __riscv_vse32_v_f32m1

// STALE-ARTIFACT-BASELINE: metadata key '{{.*}}low_precision_resource.performance_baseline'{{.*}}'scalar-c-reference/product-reduction-dequant-clamp-packed-i4-v1' but was 'scalar-c-reference/product-reduction-dequant-packed-i4-v1'

// STALE-ARTIFACT-MEASUREMENT: metadata key '{{.*}}low_precision_resource.remediation_measurement_evidence'{{.*}}'gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json' but was 'gate4-packed-i4-scalar-epilogue-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json'

// STALE-POLICY-DISPATCH-PREFERENCE: metadata key '{{.*}}low_precision_resource.selected_dispatch_preference'{{.*}}'not-performance-preferred' but was 'performance-preferred'

// STALE-POLICY-CORRECTNESS-FALLBACK: metadata key '{{.*}}low_precision_resource.correctness_fallback_path_selected'{{.*}}'true' but was 'false'
