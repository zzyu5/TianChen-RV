// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/vector-compare-rhs-load/s//script-derived-mask-producer/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-PRODUCER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-mask-tail-policy-route-family-plan.v1/s//rvv-script-derived-mask-tail-policy-plan.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-TAIL-PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/computed-mask select mask.tail policy/s//script-derived mask-tail owner/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-TAIL-OWNER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// vector-select slice. Compare operands, true/false vector operands, output,
// runtime n/AVL, typed vector config, and tail/mask policy are all explicit
// RVV-body facts. The provider may choose RVV target leaves only after this
// body is realized into generic compare/select/store structure.

module {
  tcrv.exec.kernel @pre_realized_body_computed_mask_select_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_computed_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:true_value", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:false_value", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_computed_mask_select {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_mask_select
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[TRUE:.*]] = tcrv_rvv.load
// REALIZED: %[[FALSE:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !tcrv_rvv.mask<i32, "m1">
// REALIZED: %[[SELECTED:.*]] = tcrv_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_mask_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.select"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-vector-select"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,true_value,false_value,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_mask_select.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_mask_select.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.mask_tail_policy_owner", value = "computed-mask select mask/tail policy"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-select-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-select-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.select_layout", value = "select-true-value-when-mask-else-false-value"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-mask-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-mask-select-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,true_value,false_value,out,n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-select-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-select-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_mask_select.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_mask_select.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_select_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1
// HEADER: tianchenrv.rvv.mask_tail_policy_owner: computed-mask select mask/tail policy
// HEADER: void tcrv_emitc_pre_realized_body_computed_mask_select_kernel_pre_realized_body_rvv_computed_mask_select(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);

// STALE-CMS-MASK-PRODUCER: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-PRODUCER: tcrv_rvv.computed_mask_select_mask_producer_source
// STALE-CMS-MASK-PRODUCER-SAME: must mirror
// STALE-CMS-MASK-PRODUCER-SAME: script-derived-mask-producer

// STALE-CMS-MASK-TAIL-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-TAIL-PLAN: tcrv_rvv.mask_tail_policy_route_family_plan
// STALE-CMS-MASK-TAIL-PLAN-SAME: must mirror
// STALE-CMS-MASK-TAIL-PLAN-SAME: rvv-script-derived-mask-tail-policy-plan.v1

// STALE-CMS-MASK-TAIL-OWNER: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-TAIL-OWNER: tcrv_rvv.mask_tail_policy_owner
// STALE-CMS-MASK-TAIL-OWNER-SAME: must mirror
// STALE-CMS-MASK-TAIL-OWNER-SAME: script-derived mask-tail owner
