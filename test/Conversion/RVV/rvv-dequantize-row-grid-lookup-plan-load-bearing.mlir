// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/codebook_entry_lanes = 8 : i64/codebook_entry_lanes = 11 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTATED
// RUN: sed 's/, codebook_entry_lanes = 8 : i64//' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=NOENTRY

// JUDGMENT experiment for phase-4 (family #4): the GridLookup MechanismPlan is LOAD-BEARING
// (the dequant-row grid emitter READS plan.*, it is not dead data the emitter re-derives
// around).
//
// Phase-4 introduced weft::GridLookupPlan (Support) + the FormulaProvider
// gridLookupPlanFromFacts (RVVGearboxSchedule.h) and rewired the IQ grid-table dequant emit
// (iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s) so the ONLY path from the decode_core descriptor to
// the emitted C is
//   decode_core facts -> gridLookupPlanFromFacts -> GridLookupPlan -> emit.
// The provider FOLDS the fail-closed GridDecodePlan registry THROUGH itself (it consults
// lookupGridDecodePlan for legality, so the dequant-row head shares the SAME closed-set
// authority the block-dot head uses). The dispatch tests plan.mechanism == GridLookup, NOT
// the format string; the per-format owned body is selected by plan.leaf. This test proves
// the plan is really consumed by mutating a re-packaged plan field and observing the emit
// change:
//
//   plan.entryLanes -- the grid ENTRY byte-width g-axis geometry the 律2 migration retired
//   into the codebook_entry_lanes descriptor. For iq2_xs (a grid-of-8 leaf) it is 8; the
//   emitter reads it as the grid-entry pointer stride AND the narrow unit-stride vle8 vl.
//   Mutating the descriptor 8 -> 11 changes the emitted vl literal 8 -> 11 (a byte-diff the
//   mutation cannot fake away), proving the emit CONSUMES plan.entryLanes rather than baking
//   the constant. With the descriptor ABSENT the op FAILS CLOSED (the GridLookupPlan legality
//   / the grid g-axis gate is real, never value_or self-supplied).
//
// Byte-exact reproduce-current is proved SEPARATELY (the 5 golden grid dequant lits are
// unchanged, md5-identical to the pre-phase-4 emit); this file is the load-bearing half. The
// input is the CONSTRUCTED typed region directly (NOT the abstract weft_rvv.dequantize_row)
// so the mutated decode_core attrs are consumed verbatim.

module {
  weft.exec.kernel @dequant_iq2_xs_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq2_xs attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq2_xs, sew = 32 : i64, source_kernel = "dequant_iq2_xs_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "iq2_xs", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 74 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "iq2_xs", dequant_mechanism = "grid-lookup", grid_decode_leaf = "iq2-xs", qk = 256 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 74 : i64, codebook_entry_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (codebook_entry_lanes = 8): plan.entryLanes (8) is the grid-entry pointer stride
// and the grid-entry unit-stride vle8 vl. The sign-plane pointer is offset by
// (selector * entryLanes) and the grid entry is loaded by a narrow vle8 whose vl is "8".
// CHECK: emitc.func @weft_emitc_dequant_iq2_xs_kernel_dequant_iq2_xs(
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
// CHECK-NEXT: %[[VL:.*]] = literal "8" : !emitc.opaque<"size_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf2
// CHECK-NEXT: call_opaque "__riscv_vle8_v_i8mf2"(%{{.*}}, %[[VL]])

// MUTATED (descriptor 8 -> 11): the RE-PACKAGED plan.entryLanes reaches emit -- the SAME
// grid-entry load's vl becomes "11", proving the emit CONSUMES plan.entryLanes.
// MUTATED: add %{{.*}}, %{{.*}} : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int8_t">>
// MUTATED-NEXT: %[[VL2:.*]] = literal "11" : !emitc.opaque<"size_t">
// MUTATED-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf2
// MUTATED-NEXT: call_opaque "__riscv_vle8_v_i8mf2"(%{{.*}}, %[[VL2]])

// NOENTRY (descriptor ABSENT): fail closed with a named diagnostic -- the grid g-axis
// geometry is never value_or self-supplied (anti-decoration).
// NOENTRY: selected grid leaf requires construction-owned codebook_entry_lanes geometry
