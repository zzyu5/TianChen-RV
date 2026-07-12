// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"/s//weft_rvv.mask_role", value = "script-derived-mask-role"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK-ROLE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"/s//weft_rvv.masked_passthrough_layout", value = "script-derived-passthrough-layout"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK-PASSTHROUGH

// Pre-realized masked selected-body input. The RVV plugin must consume the
// explicit mask-source and passthrough facts into typed weft_rvv compare and
// masked_binary structure before the provider/common EmitC/target path can
// consume it.

module {
  weft.exec.kernel @pre_realized_body_masked_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_masked_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_masked_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", masked_passthrough = "passthrough-vector-preserves-inactive-lanes", memory_form = "masked-vector-rhs-load", op_kind = "masked_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_masked_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_masked_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_add
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "eq"
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_binary %[[MASK]], %[[LHS]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "add"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_masked_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_binary"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-add-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-add-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "weft_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-passthrough-vector"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_masked_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-masked-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: weft.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: weft.rvv.target_leaf_profile: rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated
// HEADER-DAG: weft.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: weft.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_add.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-add-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-add-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void weft_emitc_pre_realized_body_masked_add_kernel_pre_realized_body_rvv_masked_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// STALE-MASK-ROLE: RVV materialized EmitC target artifact bridge failed
// STALE-MASK-ROLE: weft_rvv.mask_role
// STALE-MASK-ROLE-SAME: must mirror
// STALE-MASK-ROLE-SAME: script-derived-mask-role

// STALE-MASK-PASSTHROUGH: RVV materialized EmitC target artifact bridge failed
// STALE-MASK-PASSTHROUGH: weft_rvv.masked_passthrough_layout
// STALE-MASK-PASSTHROUGH-SAME: must mirror
// STALE-MASK-PASSTHROUGH-SAME: script-derived-passthrough-layout
