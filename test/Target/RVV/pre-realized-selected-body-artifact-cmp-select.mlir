// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized compare/select selected-body input. The RVV plugin must consume
// explicit predicate and select-layout facts into generic tcrv_rvv compare and
// select structure before the provider/common EmitC/target path can consume it.
// The direct emission-plan runs above prove the production route-entry bridge
// performs that realization before target artifact/header export without a
// separate selected-lowering-boundary materialization pass.

module {
  tcrv.exec.kernel @pre_realized_body_cmp_select_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_cmp_select attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_cmp_select {origin = "rvv-plugin", policy = "pre-realized-selected-body-cmp-select-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-cmp-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_compare_select_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmp_select
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "eq"
// REALIZED-SAME: -> !tcrv_rvv.mask<i32, "m1">
// REALIZED: %[[SELECTED:.*]] = tcrv_rvv.select %[[MASK]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_compare_select_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "cmp_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.select"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "i32"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:cmp_select.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:cmp_select.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|select-true-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|select-false-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "tcrv_rvv.plain_compare_select_route_family_plan", value = "rvv-plain-compare-select-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-compare-select-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-compare-select-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.select_layout", value = "select-lhs-when-mask-else-rhs"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-cmp-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmp_select

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_cmp_select
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-cmp-select-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.element_type: i32
// HEADER: tianchenrv.rvv.compare_predicate_kind: eq
// HEADER: tianchenrv.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: tianchenrv.rvv.mask_memory_form: compare-produced-mask
// HEADER: tianchenrv.rvv.select_layout: select-lhs-when-mask-else-rhs
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-plain-compare-select-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:cmp_select.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:cmp_select.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|select-true-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|select-false-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: tianchenrv.rvv.plain_compare_select_route_family_plan: rvv-plain-compare-select-route-family-plan.v1
// HEADER: void tcrv_emitc_pre_realized_body_cmp_select_kernel_pre_realized_body_rvv_cmp_select(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
