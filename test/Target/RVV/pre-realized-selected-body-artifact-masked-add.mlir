// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"/s//tcrv_rvv.mask_role", value = "script-derived-mask-role"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK-ROLE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"/s//tcrv_rvv.masked_passthrough_layout", value = "script-derived-passthrough-layout"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK-PASSTHROUGH

// Pre-realized masked selected-body input. The RVV plugin must consume the
// explicit mask-source and passthrough facts into typed tcrv_rvv compare and
// masked_binary structure before the provider/common EmitC/target path can
// consume it.

module {
  tcrv.exec.kernel @pre_realized_body_masked_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_masked_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_masked_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", masked_passthrough = "passthrough-vector-preserves-inactive-lanes", memory_form = "masked-vector-rhs-load", op_kind = "masked_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_masked_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_masked_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_add
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "eq"
// REALIZED: %[[SUM:.*]] = tcrv_rvv.masked_binary %[[MASK]], %[[LHS]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "add"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_masked_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_binary"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-add-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-add-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "tcrv_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-passthrough-vector"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_masked_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-masked-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: tianchenrv.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated
// HEADER-DAG: tianchenrv.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_add.v1
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-add-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-add-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void tcrv_emitc_pre_realized_body_masked_add_kernel_pre_realized_body_rvv_masked_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// STALE-MASK-ROLE: RVV materialized EmitC target artifact bridge failed
// STALE-MASK-ROLE: tcrv_rvv.mask_role
// STALE-MASK-ROLE-SAME: must mirror
// STALE-MASK-ROLE-SAME: script-derived-mask-role

// STALE-MASK-PASSTHROUGH: RVV materialized EmitC target artifact bridge failed
// STALE-MASK-PASSTHROUGH: tcrv_rvv.masked_passthrough_layout
// STALE-MASK-PASSTHROUGH-SAME: must mirror
// STALE-MASK-PASSTHROUGH-SAME: script-derived-passthrough-layout
