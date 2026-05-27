// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-scalar-broadcast-elementwise-route-family-plan.v1/s//rvv-script-derived-scalar-broadcast-plan.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lhs,rhs_scalar,out,n/s//lhs,out,rhs_scalar,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-ABI

// Pre-realized selected-body input for one bounded Stage2 vector-scalar add.
// The RVV plugin must realize the explicit RHS scalar ABI value into generic
// tcrv_rvv.splat dataflow before the provider route/common EmitC/target path
// may construct the scalar-broadcast route.

module {
  tcrv.exec.kernel @pre_realized_body_scalar_broadcast_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_scalar_broadcast_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs_scalar, %out, %n {lmul = "m1", memory_form = "rhs-scalar-broadcast", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_scalar_broadcast_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-scalar-broadcast-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-scalar-broadcast-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_scalar_broadcast_add
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.splat
// REALIZED: tcrv_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "scalar_broadcast_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "rhs-scalar-broadcast"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:scalar_broadcast_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:scalar_broadcast_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|scalar-broadcast-lhs-call|header-mirror;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", value = "rvv-scalar-broadcast-elementwise-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-scalar-broadcast-elementwise-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs:signed-e32m1,rhs_scalar:i32,result:signed-e32m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-scalar-broadcast-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_scalar_broadcast_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_scalar_broadcast_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,out,n
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|scalar-broadcast-lhs-call|header-mirror;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: tianchenrv.rvv.scalar_broadcast_elementwise_route_family_plan: rvv-scalar-broadcast-elementwise-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs:signed-e32m1,rhs_scalar:i32,result:signed-e32m1
// HEADER: void tcrv_emitc_pre_realized_body_scalar_broadcast_add_kernel_pre_realized_body_rvv_scalar_broadcast_add(const int32_t *lhs, int32_t rhs_scalar, int32_t *out, size_t n);

// STALE-SB-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-SB-PLAN: tcrv_rvv.scalar_broadcast_elementwise_route_family_plan
// STALE-SB-PLAN-SAME: must mirror
// STALE-SB-PLAN-SAME: rvv-script-derived-scalar-broadcast-plan.v1

// STALE-SB-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-SB-ABI: tcrv_rvv.runtime_abi_order
// STALE-SB-ABI-SAME: must mirror
// STALE-SB-ABI-SAME: lhs,out,rhs_scalar,n
