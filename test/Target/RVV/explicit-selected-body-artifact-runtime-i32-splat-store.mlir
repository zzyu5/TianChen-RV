// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @explicit_selected_body_runtime_i32_splat_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_runtime_i32_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-i32-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-runtime-i32-splat-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-i32-splat-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_runtime_i32_splat_store, sew = 32 : i64, source_kernel = "explicit_selected_body_runtime_i32_splat_store_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_runtime_i32_splat_store {origin = "rvv-plugin", policy = "explicit-runtime-i32-splat-store-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-runtime-i32-splat-store-fallback"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_i32_splat_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.splat"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-splat-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "rhs_scalar,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_i32_splat_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_i32_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_scalar_splat_store_route_family_plan", value = "rvv-runtime-scalar-splat-store-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-runtime-scalar-splat-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,rhs_scalar:i32,result:signed-e32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-i32-splat-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_runtime_i32_splat_store

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_runtime_i32_splat_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-i32-splat-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_parameter[0]: int32_t rhs_scalar role=rhs-scalar-value ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[1]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_order: rhs_scalar,out,n
// HEADER: tianchenrv.rvv.memory_form: runtime-scalar-splat-store
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-runtime-scalar-splat-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_i32_splat_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_i32_splat_store.v1;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void tcrv_emitc_explicit_selected_body_runtime_i32_splat_store_kernel_explicit_selected_body_rvv_runtime_i32_splat_store(int32_t rhs_scalar, int32_t *out, size_t n);
