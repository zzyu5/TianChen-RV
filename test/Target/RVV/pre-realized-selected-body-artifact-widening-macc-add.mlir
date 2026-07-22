// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// multiply-accumulate slice. The RVV plugin must derive i16mf2 lhs/rhs,
// i32m1 accumulator/result, relation, policy, route, and ABI facts from typed
// body/config facts.

module {
  weft.exec.kernel @pre_realized_body_widening_macc_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_widening_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-macc-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_macc_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", memory_form = "unit-stride-widening-macc", op_kind = "signed_widening_macc_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-widening-multiply-accumulate-result-to-output-buffer", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_widening_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-macc-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-macc-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m1"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widening_macc_add
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: %[[SUM:.*]] = weft_rvv.widening_macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "signed_widening_macc_add"
// REALIZED-SAME: macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1"
// REALIZED-SAME: result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED-NOT: weft_rvv.typed_widening_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widening_macc_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_widening_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_widening_macc_add_kernel_pre_realized_body_rvv_widening_macc_add(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);
