// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_cmp_select_sle_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmp_select_sle attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select-sle:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select-sle:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select-sle:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select-sle:n", role = "runtime-element-count"} : index
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmp_select_sle {origin = "rvv-plugin", policy = "pre-realized-selected-body-cmp-select-sle-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-cmp-select-sle-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmp_select_sle
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "cmp_select"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:cmp_select.v1"}
// PLAN-SAME: {key = "weft_rvv.plain_compare_select_route_family_plan", value = "rvv-plain-compare-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-compare-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-compare-select-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.select_layout", value = "select-lhs-when-mask-else-rhs"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-cmp-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmp_select_sle

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmp_select_sle
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.memory_form: vector-rhs-load
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:cmp_select.v1
// HEADER: weft.rvv.plain_compare_select_route_family_plan: rvv-plain-compare-select-route-family-plan.v1
// HEADER: void weft_emitc_pre_realized_body_cmp_select_sle_kernel_pre_realized_body_rvv_cmp_select_sle(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
