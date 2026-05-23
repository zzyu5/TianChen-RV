// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Explicit selected-body input for one bounded Stage 2 signed computed-mask
// runtime-strided-input widening dot-product reduction slice. The generic
// tcrv_rvv body carries compare mask provenance, element-strided i16mf2 source
// loads, scalar seed/result, runtime VL/AVL, and provider-derived route facts.

module {
  tcrv.exec.kernel @explicit_masked_strided_dot_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_explicit_masked_strided_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:lhs", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:rhs", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_explicit_masked_strided_dot, sew = 32 : i64, source_kernel = "explicit_masked_strided_dot_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %a = tcrv_rvv.strided_load %lhs, %lhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.strided_load %rhs, %rhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_widening_dot_reduce %mask, %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_masked_widening_dot_reduce_add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_explicit_masked_strided_dot {origin = "rvv-plugin", policy = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_strided_input_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_widening_dot_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_strided_wdot.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_strided_wdot.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask;dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16;dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_zeroing_requirement", value = "masked-widening-products-zero-inactive-lanes-before-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.lhs_stride_source", value = "runtime_abi:lhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_stride_source", value = "runtime_abi:rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.masked_widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1_m"}
// PLAN-SAME: {key = "tcrv_rvv.strided_load_intrinsic", value = "__riscv_vlse16_v_i16mf2"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_masked_strided_dot

// HEADER: tianchenrv.rvv.selected_variant: @rvv_explicit_masked_strided_dot
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: tianchenrv.rvv.memory_form: computed-mask-strided-input-widening-dot-reduce
// HEADER: tianchenrv.rvv.strided_memory_layout: unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi
// HEADER: tianchenrv.rvv.lhs_stride_source: runtime_abi:lhs_stride
// HEADER: tianchenrv.rvv.rhs_stride_source: runtime_abi:rhs_stride
// HEADER: tianchenrv.rvv.source_memory_form: strided-load
// HEADER: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER: tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: tianchenrv.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_strided_wdot.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_strided_wdot.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask;dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16;dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr
// HEADER: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void tcrv_emitc_explicit_masked_strided_dot_kernel_rvv_explicit_masked_strided_dot(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);
