// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 segment2
// deinterleave memory movement slice. The RVV plugin must realize the
// interleaved source, field0 output, field1 output, and runtime n ABI operands
// into explicit segment2_load/move/move/store/store typed structure before the
// provider may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_segment2_deinterleave_unit_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_segment2_deinterleave_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:out0", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:out1", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_segment2_deinterleave_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-segment2-deinterleave-unit-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-segment2-deinterleave-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_segment2_deinterleave_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[FIELD0:.*]], %[[FIELD1:.*]] = weft_rvv.segment2_load
// REALIZED-SAME: field0_role = "segment-field0-output-buffer"
// REALIZED-SAME: field1_role = "segment-field1-output-buffer"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: source_memory_form = "segment2-interleaved-unit-stride-load"
// REALIZED: %[[MOVED0:.*]] = weft_rvv.move %[[FIELD0]], %[[VL]]
// REALIZED-SAME: kind = "copy"
// REALIZED: %[[MOVED1:.*]] = weft_rvv.move %[[FIELD1]], %[[VL]]
// REALIZED-SAME: kind = "copy"
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
// REALIZED-NOT: weft_rvv.indexed_store
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_segment2_deinterleave_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_segment2_deinterleave_unit_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_segment2_deinterleave_unit_store
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_segment2_deinterleave_unit_store_kernel_pre_realized_body_rvv_segment2_deinterleave_unit_store(const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
