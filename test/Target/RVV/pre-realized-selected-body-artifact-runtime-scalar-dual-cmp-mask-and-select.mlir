// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_dual_cmp_mask_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_rvv_dual_cmp_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs_a = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_a", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_a = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_a", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_b", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_b = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_b", role = "rhs-secondary-scalar-value"} : i32
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body %cmp_lhs_a, %rhs_scalar_a, %cmp_lhs_b, %rhs_scalar_b, %true_value, %false_value, %out, %n {lmul = "m1", mask_composition = "and", mask_memory_form = "composed-compare-produced-mask", mask_role = "predicate-mask-produced-by-mask-and", mask_source = "mask-and-of-two-runtime-scalar-compare-produced-masks", memory_form = "runtime-scalar-dual-cmp-mask-and-select", op_kind = "runtime_scalar_dual_cmp_mask_and_select", predicate_kind_a = "sle", predicate_kind_b = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_dual_cmp_mask_select_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_rvv_dual_cmp_mask_select {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-case"}
      weft.exec.fallback @pre_dual_cmp_mask_select_scalar {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_rvv_dual_cmp_mask_select
// REALIZED: %[[LHS_A:.*]] = weft_rvv.load
// REALIZED: %[[RHS_A:.*]] = weft_rvv.splat
// REALIZED: %[[LHS_B:.*]] = weft_rvv.load
// REALIZED: %[[RHS_B:.*]] = weft_rvv.splat
// REALIZED: %[[TRUE:.*]] = weft_rvv.load
// REALIZED: %[[FALSE:.*]] = weft_rvv.load
// REALIZED: %[[MASK_A:.*]] = weft_rvv.compare %[[LHS_A]], %[[RHS_A]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[MASK_B:.*]] = weft_rvv.compare %[[LHS_B]], %[[RHS_B]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[COMPOSED:.*]] = weft_rvv.mask_and %[[MASK_A]], %[[MASK_B]], %[[VL]]
// REALIZED-SAME: kind = "and"
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[COMPOSED]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_dual_cmp_mask_and_select"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "runtime-scalar-dual-cmp-mask-and-select"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_mask_producer_source", value = "dual-runtime-scalar-splat-compare-rhs-mask-and"}
// PLAN-SAME: {key = "weft_rvv.secondary_compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.mask_composition", value = "and"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_rvv_dual_cmp_mask_select

// HEADER: weft.rvv.selected_variant: @pre_rvv_dual_cmp_mask_select
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n
// HEADER: weft.rvv.secondary_compare_predicate_kind: sle
// HEADER: weft.rvv.mask_composition: and
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER: weft.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_select_mask_producer_source: dual-runtime-scalar-splat-compare-rhs-mask-and
// HEADER: void weft_emitc_pre_dual_cmp_mask_select_kernel_pre_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
