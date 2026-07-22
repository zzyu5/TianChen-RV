// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_cmp_select_lmul_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmp_select_lmul_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-cmp-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m2", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmp_select_lmul_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-cmp-select-lmul-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-cmp-select-lmul-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmp_select_lmul_m2
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[SELECTED:.*]] = weft_rvv.select %[[MASK]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_compare_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmp_select_lmul_m2

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmp_select_lmul_m2
// HEADER: void weft_emitc_pre_realized_body_cmp_select_lmul_m2_kernel_pre_realized_body_rvv_cmp_select_lmul_m2(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
