// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_computed_mask_select_lmul_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_mask_select_lmul_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-select-lmul-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_select_pre_realized_body %cmp_lhs, %cmp_rhs, %true_value, %false_value, %out, %n {lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-vector-select", op_kind = "computed_mask_select", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_mask_select_lmul_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-select-lmul-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-select-lmul-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[TRUE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[FALSE:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[TRUE]], %[[FALSE]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_computed_mask_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: target = @pre_realized_body_rvv_computed_mask_select_lmul_m2

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_mask_select_lmul_m2
// HEADER: void weft_emitc_pre_realized_body_computed_mask_select_lmul_m2_kernel_pre_realized_body_rvv_computed_mask_select_lmul_m2(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
