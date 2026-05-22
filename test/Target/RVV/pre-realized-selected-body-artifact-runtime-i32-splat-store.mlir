// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_runtime_i32_splat_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_runtime_i32_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-i32-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-i32-splat-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-i32-splat-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_i32_splat_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_runtime_i32_splat_store {origin = "rvv-plugin", policy = "pre-realized-runtime-i32-splat-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-runtime-i32-splat-store-fallback"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_runtime_i32_splat_store
// REALIZED: tcrv_rvv.splat
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.load
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_i32_splat_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.splat"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-splat-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "rhs_scalar,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_i32_splat_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_i32_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-runtime-scalar-splat-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-i32-splat-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_runtime_i32_splat_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_runtime_i32_splat_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-i32-splat-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: rhs_scalar,out,n
// HEADER: tianchenrv.rvv.memory_form: runtime-scalar-splat-store
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-runtime-scalar-splat-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_i32_splat_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_i32_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void tcrv_emitc_pre_realized_body_runtime_i32_splat_store_kernel_pre_realized_body_rvv_runtime_i32_splat_store(int32_t rhs_scalar, int32_t *out, size_t n);
