// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed 's/{key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "[^"]*"}/{key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "artifact-name-derived-resource-candidate"}/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE

// Explicit selected-body input for one bounded Stage 2 signed widening
// dot-product reduction slice with runtime element strides on both i16mf2
// source operands. The generic tcrv_rvv body carries stride, dot LHS/RHS,
// scalar seed/result, runtime VL/AVL, and provider-derived route facts.

module {
  tcrv.exec.kernel @explicit_strided_dot_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_explicit_strided_input_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_explicit_strided_input_dot, sew = 32 : i64, source_kernel = "explicit_strided_dot_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.strided_load %lhs, %lhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.strided_load %rhs, %rhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %sum = tcrv_rvv.widening_dot_reduce %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_widening_dot_reduce_add", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_explicit_strided_input_dot {origin = "rvv-plugin", policy = "explicit-selected-body-strided-input-widening-dot-reduce-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-strided-input-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "strided_input_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_dot_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:strided_widening_dot_reduce.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.product_emul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.accumulator_emul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.memory_form", value = "strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.runtime_abi_order", value = "lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.lhs_stride_source", value = "runtime_abi:lhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_stride_source", value = "runtime_abi:rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.strided_load_intrinsic", value = "__riscv_vlse16_v_i16mf2"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_strided_input_dot

// HEADER: tianchenrv.rvv.selected_variant: @rvv_explicit_strided_input_dot
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: tianchenrv.rvv.memory_form: strided-input-widening-dot-reduce
// HEADER: tianchenrv.rvv.strided_memory_layout: element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi
// HEADER: tianchenrv.rvv.lhs_stride_source: runtime_abi:lhs_stride
// HEADER: tianchenrv.rvv.rhs_stride_source: runtime_abi:rhs_stride
// HEADER: tianchenrv.rvv.source_memory_form: strided-load
// HEADER: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]
// HEADER: tianchenrv.rvv.low_precision_resource.product_emul: m1
// HEADER: tianchenrv.rvv.low_precision_resource.accumulator_emul: m1
// HEADER: tianchenrv.rvv.low_precision_resource.memory_form: strided-input-widening-dot-reduce
// HEADER: tianchenrv.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: tianchenrv.rvv.low_precision_resource.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: tianchenrv.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_widening_dot_reduce.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr
// HEADER: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void tcrv_emitc_explicit_strided_dot_kernel_rvv_explicit_strided_input_dot(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);

// STALE-RESOURCE: target artifact candidate validation failed
// STALE-RESOURCE-SAME: low_precision_resource.selected_candidate
// STALE-RESOURCE-SAME: provider-selected low-precision direct-contraction resource selected candidate
