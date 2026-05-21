// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_computed_mask_select_sle_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_computed_mask_select_sle attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:true_value", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:false_value", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-sle:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_computed_mask_select_sle {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-sle-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-sle-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_mask_select_sle
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[TRUE:.*]] = tcrv_rvv.load
// REALIZED: %[[FALSE:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[SELECTED:.*]] = tcrv_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_mask_select"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-vector-select"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_mask_select.v1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-mask-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select_sle

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select_sle
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_mask_select.v1
// HEADER: void tcrv_emitc_pre_realized_body_computed_mask_select_sle_kernel_pre_realized_body_rvv_computed_mask_select_sle(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
