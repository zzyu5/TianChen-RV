// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @explicit_selected_body_scalar_broadcast_mul_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_scalar_broadcast_mul attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-mul:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-mul:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-mul:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-mul:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_scalar_broadcast_mul, sew = 32 : i64, source_kernel = "explicit_selected_body_scalar_broadcast_mul_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %product = weft_rvv.binary %a, %b, %vl {kind = "mul"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_scalar_broadcast_mul {origin = "rvv-plugin", policy = "explicit-selected-body-scalar-broadcast-mul-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-scalar-broadcast-mul-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "scalar_broadcast_mul"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "rhs-scalar-broadcast"}
// PLAN-SAME: {key = "weft_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:scalar_broadcast_mul.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:scalar_broadcast_mul.v1;lhs=lhs-input-buffer:lhs:abi|materialized-load-base|scalar-broadcast-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|scalar-broadcast-rhs-call|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.scalar_broadcast_elementwise_route_family_plan", value = "rvv-scalar-broadcast-elementwise-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-scalar-broadcast-elementwise-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,result:typed-vector"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-scalar-broadcast-mul-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_scalar_broadcast_mul

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_scalar_broadcast_mul
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-mul-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,out,n
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_mul.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_mul.v1;lhs=lhs-input-buffer:lhs:abi|materialized-load-base|scalar-broadcast-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|scalar-broadcast-rhs-call|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: void weft_emitc_explicit_selected_body_scalar_broadcast_mul_kernel_explicit_selected_body_rvv_scalar_broadcast_mul(const int32_t *lhs, int32_t rhs_scalar, int32_t *out, size_t n);
