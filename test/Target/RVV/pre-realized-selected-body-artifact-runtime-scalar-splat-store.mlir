// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_runtime_scalar_splat_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_runtime_scalar_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_scalar_splat_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_runtime_scalar_splat_store {origin = "rvv-plugin", policy = "pre-realized-runtime-scalar-splat-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-runtime-scalar-splat-store-fallback"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_runtime_scalar_splat_store
// REALIZED: weft_rvv.splat
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.load
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_splat_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.splat"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "runtime-scalar-splat-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "rhs_scalar,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_splat_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "weft_rvv.runtime_scalar_splat_store_route_family_plan", value = "rvv-runtime-scalar-splat-store-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-splat-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,rhs_scalar:typed-scalar,result:typed-vector"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-splat-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_runtime_scalar_splat_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_runtime_scalar_splat_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-splat-store-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_parameter[0]: int32_t rhs_scalar role=rhs-scalar-value ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_parameter[1]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_order: rhs_scalar,out,n
// HEADER: weft.rvv.memory_form: runtime-scalar-splat-store
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-splat-store-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_splat_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,rhs_scalar:typed-scalar,result:typed-vector
// HEADER: void weft_emitc_pre_realized_body_runtime_scalar_splat_store_kernel_pre_realized_body_rvv_runtime_scalar_splat_store(int32_t rhs_scalar, int32_t *out, size_t n);
