// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// strided destination-store slice. The RVV plugin must realize compare
// lhs/rhs, active source, destination buffer, and destination byte stride ABI
// operands into explicit load/load/load/compare/masked_strided_store typed
// structure.

module {
  weft.exec.kernel @pre_realized_body_computed_masked_strided_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_masked_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      weft_rvv.typed_computed_mask_strided_store_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %dst, %n, %dst_stride_bytes {inactive_lane_policy = "preserve-old-destination", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-strided-store", op_kind = "computed_masked_strided_store", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_masked_strided_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-strided-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-strided-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_strided_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_masked_strided_store
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[SRC:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: weft_rvv.masked_strided_store %{{.*}}, %[[MASK]], %[[SRC]], %{{.*}}, %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-strided-store"
// REALIZED-SAME: stride_unit = "byte"
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.binary

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_masked_strided_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_masked_strided_store
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_computed_masked_strided_store_kernel_pre_realized_body_rvv_computed_masked_strided_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);
