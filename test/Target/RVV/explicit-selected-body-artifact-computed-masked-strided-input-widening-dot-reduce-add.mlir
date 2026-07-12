// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-contraction-family-plan-validated/s//provider_supported_mirror:rvv-script-derived-widening-dot/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:masked_strided_wdot.v1/s//rvv-route-operand-binding:script-derived-wdot.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride/s//cmp_lhs,cmp_rhs,lhs,rhs,acc,out,lhs_stride,n,rhs_stride/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32/s//vl:size_t,source:signed-e32m1,result:signed-e32m1,mask:b32/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-TYPE
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/rvv-contraction-route-family-plan.v1/s//rvv-script-derived-contraction-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONTRACTION
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"/s//weft_rvv.widening_dot_relation", value = "route-id-derived-relation"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RELATION
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_dot_source_accumulator_result_contract", value = "computed-mask-strided-source-before-skipped-source-ignored;inactive-products-zero-before-reduction;accumulator-out0-seed-carry;scalar-output-only-tail-preserve.v1"/s//weft_rvv.widening_dot_source_accumulator_result_contract", value = "metadata-derived-source-accumulator-result-contract"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONTRACT
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_dot_reduction_store_vl", value = "1"/s//weft_rvv.widening_dot_reduction_store_vl", value = "4"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-STOREVL
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/__riscv_vlse16_v_i16mf2/s//__riscv_vle16_v_i16mf2/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-STRIDE
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/__riscv_vwmul_vv_i32m1_m/s//__riscv_vwmul_vv_i32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASKPROD
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.low_precision_resource.selected_candidate", value = "[^"]*"/s//weft_rvv.low_precision_resource.selected_candidate", value = "artifact-name-derived-resource-candidate"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE

// Explicit selected-body input for one bounded Stage 2 signed computed-mask
// runtime-strided-input widening dot-product reduction slice. The generic
// weft_rvv body carries compare mask provenance, element-strided i16mf2 source
// loads, scalar seed/result, runtime VL/AVL, and provider-derived route facts.

module {
  weft.exec.kernel @explicit_masked_strided_dot_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_explicit_masked_strided_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:cmp-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:cmp-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:lhs", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:rhs", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_explicit_masked_strided_dot, sew = 32 : i64, source_kernel = "explicit_masked_strided_dot_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %cmp_rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %a = weft_rvv.strided_load %lhs, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %b = weft_rvv.strided_load %rhs, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %mask = weft_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %sum = weft_rvv.masked_widening_dot_reduce %mask, %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_masked_widening_dot_reduce_add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_explicit_masked_strided_dot {origin = "rvv-plugin", policy = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-masked-strided-input-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_strided_input_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_widening_dot_reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_strided_wdot.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_strided_wdot.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16|hdr;dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_zeroing_requirement", value = "masked-widening-products-zero-inactive-lanes-before-reduction"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[computed-mask-strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.product_emul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.accumulator_emul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.memory_form", value = "computed-mask-strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.mask_policy", value = "agnostic"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_resource.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.lhs_stride_source", value = "runtime_abi:lhs_stride"}
// PLAN-SAME: {key = "weft_rvv.rhs_stride_source", value = "runtime_abi:rhs_stride"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_source_accumulator_result_contract", value = "computed-mask-strided-source-before-skipped-source-ignored;inactive-products-zero-before-reduction;accumulator-out0-seed-carry;scalar-output-only-tail-preserve.v1"}
// PLAN-SAME: {key = "weft_rvv.masked_widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1_m"}
// PLAN-SAME: {key = "weft_rvv.strided_load_intrinsic", value = "__riscv_vlse16_v_i16mf2"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_masked_strided_dot

// HEADER: weft.rvv.selected_variant: @rvv_explicit_masked_strided_dot
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: weft.rvv.memory_form: computed-mask-strided-input-widening-dot-reduce
// HEADER: weft.rvv.strided_memory_layout: unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi
// HEADER: weft.rvv.lhs_stride_source: runtime_abi:lhs_stride
// HEADER: weft.rvv.rhs_stride_source: runtime_abi:rhs_stride
// HEADER: weft.rvv.source_memory_form: strided-load
// HEADER: weft.rvv.destination_memory_form: unit-stride-store
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[computed-mask-strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]
// HEADER: weft.rvv.low_precision_resource.product_emul: m1
// HEADER: weft.rvv.low_precision_resource.accumulator_emul: m1
// HEADER: weft.rvv.low_precision_resource.memory_form: computed-mask-strided-input-widening-dot-reduce
// HEADER: weft.rvv.low_precision_resource.mask_policy: agnostic
// HEADER: weft.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: weft.rvv.low_precision_resource.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: weft.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: weft.rvv.widening_dot_source_accumulator_result_contract: computed-mask-strided-source-before-skipped-source-ignored;inactive-products-zero-before-reduction;accumulator-out0-seed-carry;scalar-output-only-tail-preserve.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_strided_wdot.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_strided_wdot.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16|hdr;dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr;lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: void weft_emitc_explicit_masked_strided_dot_kernel_rvv_explicit_masked_strided_dot(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);

// STALE-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-BINDING: candidate weft_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-ABI: metadata key '{{.*}}runtime_abi_order'{{.*}}'cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride' but was 'cmp_lhs,cmp_rhs,lhs,rhs,acc,out,lhs_stride,n,rhs_stride'
// STALE-HEADER: metadata key '{{.*}}required_header_declarations'{{.*}}'stddef.h,stdint.h,riscv_vector.h' but was 'stddef.h,stdint.h'
// STALE-TYPE: metadata key '{{.*}}c_type_mapping'{{.*}}'vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32' but was 'vl:size_t,source:signed-e32m1,result:signed-e32m1,mask:b32'
// STALE-CONTRACTION: metadata key '{{.*}}contraction_route_family_plan'{{.*}}'rvv-contraction-route-family-plan.v1' but was 'rvv-script-derived-contraction-plan.v1'
// STALE-RELATION: metadata key '{{.*}}widening_dot_relation'{{.*}}'signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32' but was 'route-id-derived-relation'
// STALE-CONTRACT: metadata key '{{.*}}widening_dot_source_accumulator_result_contract'{{.*}}'computed-mask-strided-source-before-skipped-source-ignored;inactive-products-zero-before-reduction;accumulator-out0-seed-carry;scalar-output-only-tail-preserve.v1' but was 'metadata-derived-source-accumulator-result-contract'
// STALE-STOREVL: metadata key '{{.*}}widening_dot_reduction_store_vl'{{.*}}'1' but was '4'
// STALE-STRIDE: metadata key '{{.*}}strided_load_intrinsic'{{.*}}'__riscv_vlse16_v_i16mf2' but was '__riscv_vle16_v_i16mf2'
// STALE-MASKPROD: metadata key '{{.*}}masked_widening_product_intrinsic'{{.*}}'__riscv_vwmul_vv_i32m1_m' but was '__riscv_vwmul_vv_i32m1'
// STALE-RESOURCE: metadata key '{{.*}}low_precision_resource.selected_candidate'{{.*}}'rvv-low-precision-direct-contraction-resource-candidate.v1[computed-mask-strided-input-widening-dot-reduce-add,i16mf2-i32m1,u1]' but was 'artifact-name-derived-resource-candidate'
