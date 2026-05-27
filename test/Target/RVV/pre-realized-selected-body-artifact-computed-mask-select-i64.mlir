// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_computed_mask_select_i64_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_computed_mask_select_i64 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:true_value", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:false_value", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-i64:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 64 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_computed_mask_select_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-i64-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_mask_select_i64
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[TRUE:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[FALSE:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !tcrv_rvv.mask<i64, "m1">
// REALIZED: %[[SELECTED:.*]] = tcrv_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-vector-select"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,true_value,false_value,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_mask_select.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-select-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-select-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.select_layout", value = "select-true-value-when-mask-else-false-value"}
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select_i64

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select_i64
// HEADER: tianchenrv.rvv.element_type: i64
// HEADER: tianchenrv.rvv.sew: 64
// HEADER: tianchenrv.rvv.lmul: m1
// HEADER: tianchenrv.rvv.compare_predicate_kind: slt
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-select-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-select-plan-validated
// HEADER: tianchenrv.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_select_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector
// HEADER: void tcrv_emitc_pre_realized_body_computed_mask_select_i64_kernel_pre_realized_body_rvv_computed_mask_select_i64(const int64_t *cmp_lhs, const int64_t *cmp_rhs, const int64_t *true_value, const int64_t *false_value, int64_t *out, size_t n);
