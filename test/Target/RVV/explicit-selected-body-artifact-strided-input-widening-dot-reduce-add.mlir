// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/{key = "weft_rvv.low_precision_resource.selected_candidate", value = "[^"]*"}/{key = "weft_rvv.low_precision_resource.selected_candidate", value = "artifact-name-derived-resource-candidate"}/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE

// Explicit selected-body input for one bounded Stage 2 signed widening
// dot-product reduction slice with runtime element strides on both i16mf2
// source operands. The generic weft_rvv body carries stride, dot LHS/RHS,
// scalar seed/result, runtime VL/AVL, and provider-derived route facts.

module {
  weft.exec.kernel @explicit_strided_dot_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_explicit_strided_input_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-input-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_explicit_strided_input_dot, sew = 32 : i64, source_kernel = "explicit_strided_dot_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.strided_load %lhs, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %b = weft_rvv.strided_load %rhs, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %sum = weft_rvv.widening_dot_reduce %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_widening_dot_reduce_add", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_explicit_strided_input_dot {origin = "rvv-plugin", policy = "explicit-selected-body-strided-input-widening-dot-reduce-add-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-strided-input-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "strided_input_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_dot_reduce"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:strided_widening_dot_reduce.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.product_emul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.accumulator_emul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.memory_form", value = "strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.runtime_abi_order", value = "lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.lhs_stride_source", value = "runtime_abi:lhs_stride"}
// PLAN-SAME: {key = "weft_rvv.rhs_stride_source", value = "runtime_abi:rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "weft_rvv.strided_load_intrinsic", value = "__riscv_vlse16_v_i16mf2"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_strided_input_dot

// HEADER: weft.rvv.selected_variant: @rvv_explicit_strided_input_dot
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: weft.rvv.memory_form: strided-input-widening-dot-reduce
// HEADER: weft.rvv.strided_memory_layout: element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi
// HEADER: weft.rvv.lhs_stride_source: runtime_abi:lhs_stride
// HEADER: weft.rvv.rhs_stride_source: runtime_abi:rhs_stride
// HEADER: weft.rvv.source_memory_form: strided-load
// HEADER: weft.rvv.destination_memory_form: unit-stride-store
// HEADER: weft.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]
// HEADER: weft.rvv.low_precision_resource.product_emul: m1
// HEADER: weft.rvv.low_precision_resource.accumulator_emul: m1
// HEADER: weft.rvv.low_precision_resource.memory_form: strided-input-widening-dot-reduce
// HEADER: weft.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: weft.rvv.low_precision_resource.runtime_abi_order: lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: weft.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_widening_dot_reduce.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_widening_dot_reduce.v1;lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void weft_emitc_explicit_strided_dot_kernel_rvv_explicit_strided_input_dot(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);

// STALE-RESOURCE: metadata key '{{.*}}low_precision_resource.selected_candidate'{{.*}}'rvv-low-precision-direct-contraction-resource-candidate.v1[strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]' but was 'artifact-name-derived-resource-candidate'
