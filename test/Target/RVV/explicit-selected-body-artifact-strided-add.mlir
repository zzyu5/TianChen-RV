// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 strided
// memory form. The RVV route authority is the generic typed weft_rvv body:
// runtime stride SSA values are explicit ABI operands, and the provider derives
// target strided load/store intrinsics after body/config/runtime validation.

module {
  weft.exec.kernel @explicit_selected_body_strided_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_strided_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:rhs-stride", role = "rhs-input-stride"} : index
      %out_stride = weft_rvv.runtime_abi_value {c_name = "out_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:out-stride", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_strided_add, sew = 32 : i64, source_kernel = "explicit_selected_body_strided_add_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.strided_load %lhs, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.strided_load %rhs, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.strided_store %out, %sum, %out_stride, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_strided_add {origin = "rvv-plugin", policy = "explicit-selected-body-strided-add-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-strided-add-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "strided_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "strided-load-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:strided_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:strided_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load-base|binary-lhs-call|hdr;rhs=rhs-input-buffer:rhs:abi|rhs-load-base|binary-rhs-call|hdr;out=output-buffer:out:abi|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|lhs-load-stride|lhs-byte-addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|rhs-load-stride|rhs-byte-addr|hdr;out_stride=output-stride:out_stride:abi|store-stride|out-byte-addr|hdr"}
// PLAN-SAME: {key = "weft_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-strided-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-strided-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs:element-strided-typed-vector,rhs:element-strided-typed-vector,result:element-strided-typed-vector"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "element-strided-lhs-rhs-output-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "strided-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-strided-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_strided_add

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_strided_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-strided-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: weft.rvv.runtime_abi_order: lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride
// HEADER-DAG: weft.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: weft.rvv.target_leaf_profile: rvv-v1-typed-strided-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-strided-elementwise-arithmetic-plan-validated
// HEADER-DAG: weft.rvv.source_memory_form: strided-load
// HEADER-DAG: weft.rvv.destination_memory_form: strided-store
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,lhs:element-strided-typed-vector,rhs:element-strided-typed-vector,result:element-strided-typed-vector
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_add.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load-base|binary-lhs-call|hdr;rhs=rhs-input-buffer:rhs:abi|rhs-load-base|binary-rhs-call|hdr;out=output-buffer:out:abi|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|lhs-load-stride|lhs-byte-addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|rhs-load-stride|rhs-byte-addr|hdr;out_stride=output-stride:out_stride:abi|store-stride|out-byte-addr|hdr
// HEADER: void weft_emitc_explicit_selected_body_strided_add_kernel_explicit_selected_body_rvv_strided_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride, size_t out_stride);
