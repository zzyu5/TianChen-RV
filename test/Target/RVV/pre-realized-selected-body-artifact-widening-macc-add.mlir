// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// multiply-accumulate slice. The RVV plugin must derive i16mf2 lhs/rhs,
// i32m1 accumulator/result, relation, policy, route, and ABI facts from typed
// body/config facts.

module {
  tcrv.exec.kernel @pre_realized_body_widening_macc_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_widening_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_macc_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", memory_form = "unit-stride-widening-macc", op_kind = "signed_widening_macc_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-widening-multiply-accumulate-result-to-output-buffer", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_widening_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-macc-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-macc-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_widening_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m1"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widening_macc_add
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[ACC:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[SUM:.*]] = tcrv_rvv.widening_macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "signed_widening_macc_add"
// REALIZED-SAME: macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1"
// REALIZED-SAME: result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"
// REALIZED-SAME: !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED-NOT: tcrv_rvv.typed_widening_macc_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_macc"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_result_layout", value = "store-widening-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_relation", value = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widening_macc_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_widening_macc_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-macc-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.widening_macc_relation: signed-i16mf2xi16mf2-plus-i32m1-to-i32m1
// HEADER: void tcrv_emitc_pre_realized_body_widening_macc_add_kernel_pre_realized_body_rvv_widening_macc_add(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);
