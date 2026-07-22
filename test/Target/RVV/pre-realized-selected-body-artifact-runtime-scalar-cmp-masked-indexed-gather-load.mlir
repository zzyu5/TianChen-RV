// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 runtime scalar
// compare plus masked indexed gather-load slice. The RVV plugin must realize
// lhs load, rhs scalar splat, old-destination passthrough load, index load,
// compare-produced mask, masked indexed load, and final store before the route
// provider may construct the target artifact.

module {
  weft.exec.kernel @pre_realized_body_rt_scalar_cmidx_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_rt_scalar_cmidx_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body %lhs, %rhs_scalar, %src, %index, %dst, %n {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-indexed-gather-load-unit-store", offset_unit = "element", op_kind = "runtime_scalar_cmp_masked_indexed_gather_load_unit_store", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_rt_scalar_cmidx_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-gather-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[OLD_DST:.*]] = weft_rvv.load
// REALIZED: %[[INDICES:.*]] = weft_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[LOADED:.*]] = weft_rvv.masked_indexed_load %{{.*}}, %[[INDICES]], %[[MASK]], %[[OLD_DST]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED-SAME: memory_form = "masked-indexed-load"
// REALIZED-SAME: offset_unit = "element"
// REALIZED: weft_rvv.store %{{.*}}, %[[LOADED]], %[[VL]]
// REALIZED-NOT: weft_rvv.load %{{.*}}rhs_scalar
// REALIZED-NOT: weft_rvv.masked_segment2_load
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.binary

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_rt_scalar_cmidx_load

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_rt_scalar_cmidx_load
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_rt_scalar_cmidx_load_kernel_pre_realized_body_rvv_rt_scalar_cmidx_load(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);

