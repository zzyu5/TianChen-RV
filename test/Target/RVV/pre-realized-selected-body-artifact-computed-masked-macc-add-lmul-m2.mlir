// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_computed_masked_macc_add_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_masked_macc_add_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:lhs-payload", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:rhs-payload", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_macc_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-macc", op_kind = "computed_masked_macc_add", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_masked_macc_add_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-macc-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-macc-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: sew = 32
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_masked_macc_add_m2
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_macc %[[MASK]], %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.macc
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.masked_store
// REALIZED-NOT: weft_rvv.masked_load
// REALIZED-NOT: weft_rvv.typed_computed_mask_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_masked_macc_add_m2

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_masked_macc_add_m2
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_computed_masked_macc_add_m2_kernel_pre_realized_body_rvv_computed_masked_macc_add_m2(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);
