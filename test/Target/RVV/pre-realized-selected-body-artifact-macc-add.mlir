// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized macc selected-body input. The RVV plugin must consume explicit
// accumulator role/layout facts into typed weft_rvv accumulator load and macc
// structure before the existing provider/common EmitC/target path can consume it.

module {
  weft.exec.kernel @pre_realized_body_macc_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-macc-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-macc-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-macc-add:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-macc-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-macc-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_macc_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "macc_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-macc-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-macc-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_macc_add
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[SUM:.*]] = weft_rvv.macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.macc"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:macc_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.plain_macc_route_family_plan", value = "rvv-plain-macc-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-macc-add-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/rhs/acc:typed-vector,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.macc_arithmetic_kind", value = "add"}
// PLAN-SAME: {key = "weft_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_macc_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-macc-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-plain-macc-add-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-plain-macc-add-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:macc_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.plain_macc_route_family_plan: rvv-plain-macc-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,lhs/rhs/acc:typed-vector,result:typed-vector
// HEADER: void weft_emitc_pre_realized_body_macc_add_kernel_pre_realized_body_rvv_macc_add(const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);
