// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-plain-compare-select-plan-validated/s//provider_supported_mirror:rvv-script-derived-plain-compare-select/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:cmp_select.v1/s//rvv-route-operand-binding:script-derived-cmp-select.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs,rhs,out,n/s//lhs,out,rhs,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-TYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.compare_predicate_kind", value = "eq"/s//weft_rvv.compare_predicate_kind", value = "ne"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-PREDICATE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/select-lhs-when-mask-else-rhs/s//script-derived-select-layout/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CMP-LAYOUT

// Pre-realized compare/select selected-body input. The RVV plugin must consume
// explicit predicate and select-layout facts into generic weft_rvv compare and
// select structure before the provider/common EmitC/target path can consume it.
// Generated artifact and header export must run the selected lowering-boundary
// producer before emission planning; direct pre-realized route-entry authority
// for plain compare/select is fail-closed in the generated-bundle tests.

module {
  weft.exec.kernel @pre_realized_body_cmp_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmp_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmp_select {origin = "rvv-plugin", policy = "pre-realized-selected-body-cmp-select-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-cmp-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmp_select
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "eq"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m1">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "cmp_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.select"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:cmp_select.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:cmp_select.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|select-true-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|select-false-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "weft_rvv.plain_compare_select_route_family_plan", value = "rvv-plain-compare-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-compare-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-compare-select-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.select_layout", value = "select-lhs-when-mask-else-rhs"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-cmp-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmp_select

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmp_select
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-cmp-select-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.element_type: i32
// HEADER: weft.rvv.compare_predicate_kind: eq
// HEADER: weft.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.mask_memory_form: compare-produced-mask
// HEADER: weft.rvv.select_layout: select-lhs-when-mask-else-rhs
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-plain-compare-select-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:cmp_select.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:cmp_select.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|select-true-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|select-false-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: weft.rvv.plain_compare_select_route_family_plan: rvv-plain-compare-select-route-family-plan.v1
// HEADER: void weft_emitc_pre_realized_body_cmp_select_kernel_pre_realized_body_rvv_cmp_select(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// STALE-CMP-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-CMP-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-plain-compare-select

// STALE-CMP-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-BINDING: candidate weft_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-CMP-BINDING-SAME: rvv-route-operand-binding:script-derived-cmp-select.v1

// STALE-CMP-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-ABI: weft_rvv.runtime_abi_order
// STALE-CMP-ABI-SAME: must mirror
// STALE-CMP-ABI-SAME: lhs,out,rhs,n

// STALE-CMP-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-HEADER: weft_rvv.required_header_declarations
// STALE-CMP-HEADER-SAME: must mirror
// STALE-CMP-HEADER-SAME: stddef.h,stdint.h

// STALE-CMP-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-TYPE: weft_rvv.c_type_mapping
// STALE-CMP-TYPE-SAME: must mirror
// STALE-CMP-TYPE-SAME: vl:uint64_t

// STALE-CMP-PREDICATE: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-PREDICATE: weft_rvv.compare_predicate_kind
// STALE-CMP-PREDICATE-SAME: must mirror
// STALE-CMP-PREDICATE-SAME: ne

// STALE-CMP-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-CMP-LAYOUT: weft_rvv.select_layout
// STALE-CMP-LAYOUT-SAME: must mirror
// STALE-CMP-LAYOUT-SAME: script-derived-select-layout
