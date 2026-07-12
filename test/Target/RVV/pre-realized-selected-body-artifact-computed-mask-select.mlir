// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vector-compare-rhs-load/s//script-derived-mask-producer/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-PRODUCER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.compare_predicate_kind", value = "slt/s//weft_rvv.compare_predicate_kind", value = "sle/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-PREDICATE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,true_value,false_value,out,n/s//weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,false_value,true_value,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-RUNTIME-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:computed_mask_select.v1/s//rvv-route-operand-binding:script-derived-computed-mask-select.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-BINDING-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/true_value=true-value-input-buffer:true_value/s//true_value=rhs-input-buffer:true_value/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-BINDING-OPERANDS
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-computed-mask-select-plan-validated/s//provider_supported_mirror:script-derived-computed-mask-select/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-v1-typed-computed-mask-select-leaf-profile.v1/s//rvv-v1-script-derived-computed-mask-select-leaf-profile.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-TARGET-LEAF
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_role", value = "predicate-mask-produced-by-compare/s//weft_rvv.mask_role", value = "predicate-mask-input-buffer/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-ROLE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope/s//weft_rvv.mask_source", value = "route-id-derived-mask/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-SOURCE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_memory_form", value = "compare-produced-mask/s//weft_rvv.mask_memory_form", value = "unit-stride-mask-load/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-FORM
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.select_layout", value = "select-true-value-when-mask-else-false-value/s//weft_rvv.select_layout", value = "select-false-value-when-mask-else-true-value/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-SELECT-LAYOUT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.tail_policy", value = "agnostic/s//weft_rvv.tail_policy", value = "undisturbed/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-TAIL-POLICY
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_policy", value = "agnostic/s//weft_rvv.mask_policy", value = "undisturbed/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-POLICY
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-mask-tail-policy-route-family-plan.v1/s//rvv-script-derived-mask-tail-policy-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-TAIL-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/computed-mask select mask.tail policy/s//script-derived mask-tail owner/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMS-MASK-TAIL-OWNER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// vector-select slice. Compare operands, true/false vector operands, output,
// runtime n/AVL, typed vector config, and tail/mask policy are all explicit
// RVV-body facts. The provider may choose RVV target leaves only after this
// body is realized into generic compare/select/store structure.

module {
  weft.exec.kernel @pre_realized_body_computed_mask_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_mask_select {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_mask_select
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[TRUE:.*]] = weft_rvv.load
// REALIZED: %[[FALSE:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m1">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_mask_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.select"}
// PLAN-SAME: {key = "weft_rvv.tail_policy", value = "agnostic"}
// PLAN-SAME: {key = "weft_rvv.mask_policy", value = "agnostic"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-vector-select"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,true_value,false_value,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_mask_select.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_mask_select.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_owner", value = "computed-mask select mask/tail policy"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-select-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.select_layout", value = "select-true-value-when-mask-else-false-value"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-mask-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-mask-select-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,true_value,false_value,out,n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-select-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-select-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_mask_select.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_mask_select.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_select_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1
// HEADER: weft.rvv.mask_tail_policy_owner: computed-mask select mask/tail policy
// HEADER: void weft_emitc_pre_realized_body_computed_mask_select_kernel_pre_realized_body_rvv_computed_mask_select(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);

// STALE-CMS-MASK-PRODUCER: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-PRODUCER: weft_rvv.computed_mask_select_mask_producer_source
// STALE-CMS-MASK-PRODUCER-SAME: must mirror
// STALE-CMS-MASK-PRODUCER-SAME: script-derived-mask-producer

// STALE-CMS-PREDICATE: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-PREDICATE: weft_rvv.compare_predicate_kind
// STALE-CMS-PREDICATE-SAME: must mirror
// STALE-CMS-PREDICATE-SAME: sle

// STALE-CMS-RUNTIME-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-RUNTIME-ABI: weft_rvv.runtime_abi_order
// STALE-CMS-RUNTIME-ABI-SAME: cmp_lhs,cmp_rhs,false_value,true_value,out,n

// STALE-CMS-BINDING-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-BINDING-PLAN: weft_rvv.route_operand_binding_plan
// STALE-CMS-BINDING-PLAN-SAME: must mirror
// STALE-CMS-BINDING-PLAN-SAME: script-derived-computed-mask-select

// STALE-CMS-BINDING-OPERANDS: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-BINDING-OPERANDS: weft_rvv.route_operand_binding_operands
// STALE-CMS-BINDING-OPERANDS-SAME: must mirror
// STALE-CMS-BINDING-OPERANDS-SAME: true_value=rhs-input-buffer:true_value

// STALE-CMS-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-PROVIDER: weft_rvv.provider_supported_mirror
// STALE-CMS-PROVIDER-SAME: must mirror
// STALE-CMS-PROVIDER-SAME: script-derived-computed-mask-select

// STALE-CMS-TARGET-LEAF: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-TARGET-LEAF: weft_rvv.target_leaf_profile
// STALE-CMS-TARGET-LEAF-SAME: must mirror
// STALE-CMS-TARGET-LEAF-SAME: script-derived-computed-mask-select

// STALE-CMS-MASK-ROLE: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-ROLE: weft_rvv.mask_role
// STALE-CMS-MASK-ROLE-SAME: must mirror
// STALE-CMS-MASK-ROLE-SAME: predicate-mask-input-buffer

// STALE-CMS-MASK-SOURCE: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-SOURCE: weft_rvv.mask_source
// STALE-CMS-MASK-SOURCE-SAME: must mirror
// STALE-CMS-MASK-SOURCE-SAME: route-id-derived-mask

// STALE-CMS-MASK-FORM: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-FORM: weft_rvv.mask_memory_form
// STALE-CMS-MASK-FORM-SAME: must mirror
// STALE-CMS-MASK-FORM-SAME: unit-stride-mask-load

// STALE-CMS-SELECT-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-SELECT-LAYOUT: weft_rvv.select_layout
// STALE-CMS-SELECT-LAYOUT-SAME: must mirror
// STALE-CMS-SELECT-LAYOUT-SAME: select-false-value-when-mask-else-true-value

// STALE-CMS-TAIL-POLICY: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-TAIL-POLICY: weft_rvv.tail_policy
// STALE-CMS-TAIL-POLICY-SAME: must mirror
// STALE-CMS-TAIL-POLICY-SAME: undisturbed

// STALE-CMS-MASK-POLICY: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-POLICY: weft_rvv.mask_policy
// STALE-CMS-MASK-POLICY-SAME: must mirror
// STALE-CMS-MASK-POLICY-SAME: undisturbed

// STALE-CMS-MASK-TAIL-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-TAIL-PLAN: weft_rvv.mask_tail_policy_route_family_plan
// STALE-CMS-MASK-TAIL-PLAN-SAME: must mirror
// STALE-CMS-MASK-TAIL-PLAN-SAME: rvv-script-derived-mask-tail-policy-plan.v1

// STALE-CMS-MASK-TAIL-OWNER: RVV materialized EmitC target artifact bridge failed
// STALE-CMS-MASK-TAIL-OWNER: weft_rvv.mask_tail_policy_owner
// STALE-CMS-MASK-TAIL-OWNER-SAME: must mirror
// STALE-CMS-MASK-TAIL-OWNER-SAME: script-derived mask-tail owner
