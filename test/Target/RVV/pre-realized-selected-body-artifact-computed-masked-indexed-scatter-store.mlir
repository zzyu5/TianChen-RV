// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// indexed/scatter masked destination-store slice. The RVV plugin must realize
// compare lhs/rhs, unit-stride payload source, explicit index vector, and
// masked indexed no-write destination store into explicit typed structure before
// the provider may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_cmidx_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmidx_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_indexed_scatter_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %index, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-indexed-scatter-store", offset_unit = "element", op_kind = "computed_masked_indexed_scatter_store_unit_load", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmidx_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-scatter-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-scatter-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_indexed_scatter_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmidx_store
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[PAYLOAD:.*]] = weft_rvv.load
// REALIZED: %[[INDICES:.*]] = weft_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: weft_rvv.masked_indexed_store %{{.*}}, %[[INDICES]], %[[MASK]], %[[PAYLOAD]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED-SAME: index_uniqueness = "unique"
// REALIZED-SAME: memory_form = "masked-indexed-store"
// REALIZED-SAME: offset_unit = "element"
// REALIZED-NOT: weft_rvv.masked_indexed_load
// REALIZED-NOT: weft_rvv.indexed_store
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.binary

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmidx_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmidx_store
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_cmidx_store_kernel_pre_realized_body_rvv_cmidx_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
