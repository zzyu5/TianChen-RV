// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_runtime_scalar_cmp_masked_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_runtime_scalar_cmp_masked_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body %lhs, %rhs_scalar, %src, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-store", op_kind = "runtime_scalar_cmp_masked_store", predicate_kind = "sle", policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_runtime_scalar_cmp_masked_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[SRC:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: weft_rvv.masked_store %{{.*}}, %[[MASK]], %[[SRC]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-store"
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_runtime_scalar_cmp_masked_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_runtime_scalar_cmp_masked_store
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_runtime_scalar_cmp_masked_store_kernel_pre_realized_body_rvv_runtime_scalar_cmp_masked_store(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, int32_t *dst, size_t n);
