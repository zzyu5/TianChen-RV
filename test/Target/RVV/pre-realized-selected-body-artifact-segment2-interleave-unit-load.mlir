// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 segment2 interleave
// memory movement slice. The RVV plugin must realize the field0 source, field1
// source, interleaved destination, and runtime n ABI operands into explicit
// load/load/segment2_store typed structure before the provider may construct
// the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_segment2_interleave_unit_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_segment2_interleave_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:src0", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:src1", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:dst", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:n", role = "runtime-element-count"} : index
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_segment2_interleave_unit_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-segment2-interleave-unit-load-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-segment2-interleave-unit-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_segment2_interleave_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[FIELD0:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: %[[FIELD1:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.segment2_store
// REALIZED-SAME: destination_memory_form = "segment2-interleaved-unit-stride-store"
// REALIZED-SAME: field0_role = "segment-field0-input-buffer"
// REALIZED-SAME: field1_role = "segment-field1-input-buffer"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
// REALIZED-NOT: weft_rvv.segment2_load
// REALIZED-NOT: weft_rvv.indexed_store
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_segment2_interleave_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_segment2_interleave_unit_load

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_segment2_interleave_unit_load
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_segment2_interleave_unit_load_kernel_pre_realized_body_rvv_segment2_interleave_unit_load(const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);



