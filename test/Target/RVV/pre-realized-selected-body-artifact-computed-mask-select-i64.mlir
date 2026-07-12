// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_computed_mask_select_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_mask_select_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 64 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_mask_select_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-i64-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_mask_select_i64
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[TRUE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[FALSE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<i64, "m1">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "64"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-vector-select"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,true_value,false_value,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_mask_select.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-select-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.select_layout", value = "select-true-value-when-mask-else-false-value"}
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select_i64

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select_i64
// HEADER: weft.rvv.element_type: i64
// HEADER: weft.rvv.sew: 64
// HEADER: weft.rvv.lmul: m1
// HEADER: weft.rvv.compare_predicate_kind: slt
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-select-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-select-plan-validated
// HEADER: weft.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_select_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.c_type_mapping: vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector
// HEADER: void weft_emitc_pre_realized_body_computed_mask_select_i64_kernel_pre_realized_body_rvv_computed_mask_select_i64(const int64_t *cmp_lhs, const int64_t *cmp_rhs, const int64_t *true_value, const int64_t *false_value, int64_t *out, size_t n);
