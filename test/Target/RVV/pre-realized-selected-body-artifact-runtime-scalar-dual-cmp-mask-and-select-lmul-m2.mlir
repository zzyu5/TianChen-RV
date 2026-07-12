// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pr_dual_cmp_sel_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pr_rvv_dual_cmp_sel_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs_a = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:cmp_lhs_a", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_a = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:rhs_scalar_a", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:cmp_lhs_b", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_b = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:rhs_scalar_b", role = "rhs-secondary-scalar-value"} : i32
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body %cmp_lhs_a, %rhs_scalar_a, %cmp_lhs_b, %rhs_scalar_b, %true_value, %false_value, %out, %n {lmul = "m2", mask_composition = "and", mask_memory_form = "composed-compare-produced-mask", mask_role = "predicate-mask-produced-by-mask-and", mask_source = "mask-and-of-two-runtime-scalar-compare-produced-masks", memory_form = "runtime-scalar-dual-cmp-mask-and-select", op_kind = "runtime_scalar_dual_cmp_mask_and_select", predicate_kind_a = "sle", predicate_kind_b = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pr_rvv_dual_cmp_sel_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-dual-cmp-mask-and-select-lmul-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pr_rvv_dual_cmp_sel_m2
// REALIZED: %[[LHS_A:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[RHS_A:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[LHS_B:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[RHS_B:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[TRUE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[FALSE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[MASK_A:.*]] = weft_rvv.compare %[[LHS_A]], %[[RHS_A]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[MASK_B:.*]] = weft_rvv.compare %[[LHS_B]], %[[RHS_B]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[COMPOSED:.*]] = weft_rvv.mask_and %[[MASK_A]], %[[MASK_B]], %[[VL]]
// REALIZED-SAME: kind = "and"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[COMPOSED]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_dual_cmp_mask_and_select"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m2"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.secondary_compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.mask_composition", value = "and"}
// PLAN-SAME: target = @pr_rvv_dual_cmp_sel_m2

// HEADER: weft.rvv.selected_variant: @pr_rvv_dual_cmp_sel_m2
// HEADER: weft.rvv.element_type: i32
// HEADER: weft.rvv.sew: 32
// HEADER: weft.rvv.lmul: m2
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1
// HEADER: weft.rvv.c_type_mapping: vl:size_t,cmp_lhs_a:typed-vector,rhs_scalar_a:typed-scalar,cmp_lhs_b:typed-vector,rhs_scalar_b:typed-scalar,mask_a:typed-mask,mask_b:typed-mask,mask_and:typed-mask,true_false:typed-vector,result:typed-vector
// HEADER: void weft_emitc_pr_dual_cmp_sel_m2_kernel_pr_rvv_dual_cmp_sel_m2(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
