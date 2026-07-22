// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/codebook_entry_lanes = 8 : i64/codebook_entry_lanes = 11 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTATED
// RUN: sed 's/, codebook_entry_lanes = 8 : i64//' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=NOENTRY

// JUDGMENT experiment for phase-4 (family #5, the [K-10] ternary split): the TernaryDecode
// MechanismPlan is LOAD-BEARING (the dequant-row ternary emitter READS plan.*, it is not
// dead data the emitter re-derives around).
//
// Phase-4 introduced weft::TernaryDecodePlan (Support) + the FormulaProvider
// ternaryDecodePlanFromFacts (RVVGearboxSchedule.h) and rewired the ternary dequant emit
// (iq1_s/iq1_m/tq1_0/tq2_0) so the ONLY path from the decode_core descriptor to the emitted
// C is
//   decode_core facts -> ternaryDecodePlanFromFacts -> TernaryDecodePlan -> emit.
// TernaryDecode is a SEPARATE mechanism from GridLookup -- it has its OWN plan and does NOT
// consult the GridDecodePlan registry (grid != ternary RESTORED for the dequant-row head:
// the GridDecodePlan registry lumps iq1_s/iq1_m in with grid via GridFoldArith::TernaryDelta
// for the block-dot head; this head splits them out). The dispatch tests plan.mechanism ==
// TernaryDecode, NOT the format string; the per-format owned body is selected by plan.leaf.
// This test proves the plan is really consumed by mutating a re-packaged plan field and
// observing the emit change:
//
//   plan.entryLanes -- the grid ENTRY byte-width g-axis geometry the 律2 migration retired
//   into the codebook_entry_lanes descriptor. For iq1_s (a grid-of-8 iq1s_grid leaf) it is
//   8; the emitter reads it as the grid-entry unit-stride vle8 vl. Mutating the descriptor
//   8 -> 11 changes the emitted vl literal 8 -> 11 (a byte-diff the mutation cannot fake
//   away), proving the emit CONSUMES plan.entryLanes rather than baking the constant. With
//   the descriptor ABSENT the op FAILS CLOSED (the g-axis gate is real, never value_or
//   self-supplied).
//
// Byte-exact reproduce-current is proved SEPARATELY (the 4 golden ternary dequant lits are
// unchanged, md5-identical to the pre-phase-4 emit); this file is the load-bearing half. The
// input is the CONSTRUCTED typed region directly (NOT the abstract weft_rvv.dequantize_row)
// so the mutated decode_core attrs are consumed verbatim.

module {
  weft.exec.kernel @dequant_iq1_s_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq1_s attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq1_s, sew = 32 : i64, source_kernel = "dequant_iq1_s_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "iq1_s", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 50 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "iq1_s", dequant_mechanism = "ternary-decode", ternary_decode_leaf = "iq1-s", qk = 256 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 50 : i64, codebook_entry_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (codebook_entry_lanes = 8): plan.entryLanes (8) is the iq1s_grid entry pointer's
// unit-stride vle8 vl; the grid-entry pointer is cast to const int8_t and loaded with vl "8".
// CHECK: emitc.func @weft_emitc_dequant_iq1_s_kernel_dequant_iq1_s(
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
// CHECK-NEXT: %[[VL:.*]] = literal "8" : !emitc.opaque<"size_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf2
// CHECK-NEXT: call_opaque "__riscv_vle8_v_i8mf2"(%{{.*}}, %[[VL]])

// MUTATED (descriptor 8 -> 11): the RE-PACKAGED plan.entryLanes reaches emit -- the SAME
// grid-entry load's vl becomes "11", proving the emit CONSUMES plan.entryLanes.
// MUTATED: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
// MUTATED-NEXT: %[[VL2:.*]] = literal "11" : !emitc.opaque<"size_t">
// MUTATED-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf2
// MUTATED-NEXT: call_opaque "__riscv_vle8_v_i8mf2"(%{{.*}}, %[[VL2]])

// NOENTRY (descriptor ABSENT): fail closed with a named diagnostic -- the g-axis geometry is
// never value_or self-supplied (anti-decoration).
// NOENTRY: selected ternary grid leaf requires construction-owned codebook_entry_lanes geometry
