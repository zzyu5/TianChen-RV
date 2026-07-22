// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// segment2 masked load slice. The RVV plugin must realize compare lhs/rhs,
// interleaved source, field0/field1 old passthroughs, masked_segment2_load,
// and dual field stores into explicit typed structure before route construction.

module {
  weft.exec.kernel @pre_realized_body_cmseg_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmseg_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:out0", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:out1", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_segment2_load_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-segment2-load-unit-store", op_kind = "computed_masked_segment2_load_unit_store", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmseg_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-load-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_segment2_load_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmseg_load
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[OLD0:.*]] = weft_rvv.load
// REALIZED: %[[OLD1:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[FIELD0:.*]], %[[FIELD1:.*]] = weft_rvv.masked_segment2_load %{{.*}}, %[[MASK]], %[[OLD0]], %[[OLD1]], %[[VL]]
// REALIZED-SAME: field0_role = "segment-field0-output-buffer"
// REALIZED-SAME: field1_role = "segment-field1-output-buffer"
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: source_memory_form = "segment2-interleaved-unit-stride-load"
// REALIZED: weft_rvv.store %{{.*}}, %[[FIELD0]], %[[VL]]
// REALIZED: weft_rvv.store %{{.*}}, %[[FIELD1]], %[[VL]]
// REALIZED-NOT: weft_rvv.masked_indexed_load
// REALIZED-NOT: weft_rvv.masked_indexed_store
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.segment2_load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_computed_mask_segment2_load_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmseg_load

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmseg_load
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_cmseg_load_kernel_pre_realized_body_rvv_cmseg_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
