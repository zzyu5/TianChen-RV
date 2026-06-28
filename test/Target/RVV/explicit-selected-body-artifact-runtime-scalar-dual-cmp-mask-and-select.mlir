// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @explicit_dual_cmp_mask_select_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_rvv_dual_cmp_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs_a = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_a", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_a = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_a", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_b", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_b = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_b", role = "rhs-secondary-scalar-value"} : i32
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:true_value", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:false_value", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_rvv_dual_cmp_mask_select, sew = 32 : i64, source_kernel = "explicit_dual_cmp_mask_select_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_a_vec = tcrv_rvv.load %cmp_lhs_a, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_a_vec = tcrv_rvv.splat %rhs_scalar_a, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_lhs_b_vec = tcrv_rvv.load %cmp_lhs_b, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_b_vec = tcrv_rvv.splat %rhs_scalar_b, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %true_vec = tcrv_rvv.load %true_value, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %false_vec = tcrv_rvv.load %false_value, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask_a = tcrv_rvv.compare %cmp_lhs_a_vec, %rhs_a_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %mask_b = tcrv_rvv.compare %cmp_lhs_b_vec, %rhs_b_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %composed = tcrv_rvv.mask_and %mask_a, %mask_b, %vl {kind = "and"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %selected = tcrv_rvv.select %composed, %true_vec, %false_vec, %vl : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %selected, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_dual_cmp_mask_select_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_rvv_dual_cmp_mask_select {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-case"}
      tcrv.exec.fallback @explicit_dual_cmp_mask_select_scalar {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_dual_cmp_mask_and_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.select"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-dual-cmp-mask-and-select"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_mask_producer_source", value = "dual-runtime-scalar-splat-compare-rhs-mask-and"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-mask-and"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "mask-and-of-two-runtime-scalar-compare-produced-masks"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "composed-compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.secondary_compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.mask_composition", value = "and"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_rvv_dual_cmp_mask_select

// HEADER: tianchenrv.rvv.selected_variant: @explicit_rvv_dual_cmp_mask_select
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n
// HEADER: tianchenrv.rvv.secondary_compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.mask_composition: and
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER: tianchenrv.rvv.computed_mask_select_route_family_plan: rvv-computed-mask-select-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_select_mask_producer_source: dual-runtime-scalar-splat-compare-rhs-mask-and
// HEADER: void tcrv_emitc_explicit_dual_cmp_mask_select_kernel_explicit_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
