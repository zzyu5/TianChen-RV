// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-scalar-broadcast-elementwise-route-family-plan.v1/s//rvv-script-derived-scalar-broadcast-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs,rhs_scalar,out,n/s//lhs,out,rhs_scalar,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated/s//provider_supported_mirror:rvv-script-derived-scalar-broadcast/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rhs_scalar:typed-scalar/s//rhs_scalar:stale-scalar/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-TYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs=lhs-input-buffer:lhs:abi|materialized-load-base|scalar-broadcast-lhs-call|hdr/s//lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|scalar-broadcast-lhs-call|header-mirror/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SB-BINDING

// Pre-realized selected-body input for one bounded Stage2 vector-scalar add.
// The RVV plugin must realize the explicit RHS scalar ABI value into generic
// weft_rvv.splat dataflow before the provider route/common EmitC/target path
// may construct the scalar-broadcast route.

module {
  weft.exec.kernel @pre_realized_body_scalar_broadcast_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_scalar_broadcast_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_binary_pre_realized_body %lhs, %rhs_scalar, %out, %n {lmul = "m1", memory_form = "rhs-scalar-broadcast", op_kind = "add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_scalar_broadcast_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-scalar-broadcast-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-scalar-broadcast-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_scalar_broadcast_add
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.splat
// REALIZED: weft_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "scalar_broadcast_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "rhs-scalar-broadcast"}
// PLAN-SAME: {key = "weft_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:scalar_broadcast_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:scalar_broadcast_add.v1;lhs=lhs-input-buffer:lhs:abi|materialized-load-base|scalar-broadcast-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|scalar-broadcast-rhs-call|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.scalar_broadcast_elementwise_route_family_plan", value = "rvv-scalar-broadcast-elementwise-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-scalar-broadcast-elementwise-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,result:typed-vector"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-scalar-broadcast-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_scalar_broadcast_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_scalar_broadcast_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,out,n
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_add.v1;lhs=lhs-input-buffer:lhs:abi|materialized-load-base|scalar-broadcast-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|scalar-broadcast-rhs-call|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.scalar_broadcast_elementwise_route_family_plan: rvv-scalar-broadcast-elementwise-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,result:typed-vector
// HEADER: void weft_emitc_pre_realized_body_scalar_broadcast_add_kernel_pre_realized_body_rvv_scalar_broadcast_add(const int32_t *lhs, int32_t rhs_scalar, int32_t *out, size_t n);

// STALE-SB-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-SB-PLAN: weft_rvv.scalar_broadcast_elementwise_route_family_plan
// STALE-SB-PLAN-SAME: must mirror
// STALE-SB-PLAN-SAME: rvv-script-derived-scalar-broadcast-plan.v1

// STALE-SB-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-SB-ABI: weft_rvv.runtime_abi_order
// STALE-SB-ABI-SAME: must mirror
// STALE-SB-ABI-SAME: lhs,out,rhs_scalar,n

// STALE-SB-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-SB-PROVIDER: weft_rvv.provider_supported_mirror
// STALE-SB-PROVIDER-SAME: must mirror
// STALE-SB-PROVIDER-SAME: rvv-script-derived-scalar-broadcast

// STALE-SB-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-SB-TYPE: weft_rvv.c_type_mapping
// STALE-SB-TYPE-SAME: must mirror
// STALE-SB-TYPE-SAME: rhs_scalar:stale-scalar

// STALE-SB-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-SB-BINDING: weft_rvv.route_operand_binding_operands
// STALE-SB-BINDING-SAME: must mirror
// STALE-SB-BINDING-SAME: header-mirror
