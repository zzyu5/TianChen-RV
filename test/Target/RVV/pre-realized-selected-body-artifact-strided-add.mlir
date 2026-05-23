// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided add. The
// RVV plugin must realize the explicit stride ABI operands into typed
// strided_load/binary/strided_store structure before the provider route is
// allowed to construct the strided_add EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_strided_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_strided_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:n", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:rhs-stride", role = "rhs-input-stride"} : index
      %out_stride = tcrv_rvv.runtime_abi_value {c_name = "out_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:out-stride", role = "output-stride"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n, %lhs_stride, %rhs_stride, %out_stride {lmul = "m1", memory_form = "strided-load-store", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_strided_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-strided-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-strided-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_strided_add
// REALIZED: tcrv_rvv.strided_load
// REALIZED: tcrv_rvv.strided_load
// REALIZED: tcrv_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "strided_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "strided-load-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:strided_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:strided_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|rhs-load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header;lhs_stride=lhs-input-stride:lhs_stride:abi|lhs-load-stride|lhs-byte-addr|header;rhs_stride=rhs-input-stride:rhs_stride:abi|rhs-load-stride|rhs-byte-addr|header;out_stride=output-stride:out_stride:abi|store-stride|out-byte-addr|header"}
// PLAN-SAME: {key = "tcrv_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-strided-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-strided-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs:element-strided-typed-vector,rhs:element-strided-typed-vector,result:element-strided-typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "element-strided-lhs-rhs-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "strided-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-strided-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_strided_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_strided_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-strided-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: tianchenrv.rvv.runtime_abi_order: lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride
// HEADER-DAG: tianchenrv.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-strided-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-strided-elementwise-arithmetic-plan-validated
// HEADER-DAG: tianchenrv.rvv.source_memory_form: strided-load
// HEADER-DAG: tianchenrv.rvv.destination_memory_form: strided-store
// HEADER-DAG: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs:element-strided-typed-vector,rhs:element-strided-typed-vector,result:element-strided-typed-vector
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_add.v1
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|rhs-load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header;lhs_stride=lhs-input-stride:lhs_stride:abi|lhs-load-stride|lhs-byte-addr|header;rhs_stride=rhs-input-stride:rhs_stride:abi|rhs-load-stride|rhs-byte-addr|header;out_stride=output-stride:out_stride:abi|store-stride|out-byte-addr|header
// HEADER: void tcrv_emitc_pre_realized_body_strided_add_kernel_pre_realized_body_rvv_strided_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride, size_t out_stride);
