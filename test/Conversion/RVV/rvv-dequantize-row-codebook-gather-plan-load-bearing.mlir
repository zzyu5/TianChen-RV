// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/quant_byte_offset = 1 : i64/quant_byte_offset = 3 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTQS
// RUN: sed 's/qk = 32 : i64/qk = 44 : i64/g' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTSTRIP

// JUDGMENT experiment for phase-4: the CodebookGather MechanismPlan is LOAD-BEARING (the
// codebook emitter READS plan.*, it is not dead data the emitter re-derives around).
//
// Phase-4 introduced weft::CodebookGatherPlan (Support) + the FormulaProvider
// codebookGatherPlanFromFacts (RVVGearboxSchedule.h) and rewired the small-codebook emit
// (iq4_nl/iq4_xs/mxfp4/nvfp4) so the ONLY path from the decode_core descriptor to the
// emitted C is
//   decode_core facts -> codebookGatherPlanFromFacts -> CodebookGatherPlan -> emit.
// The per-format geometry re-derivation (isMx?32: ... off the format name) is RETIRED;
// the dispatch tests plan.mechanism == CodebookGather, NOT the format string. This test
// proves the plan is really consumed by mutating two DIFFERENT kinds of plan field and
// observing the emit change:
//
//   (1 qsoff)  a RE-PACKAGED geometry field: quant_byte_offset 1 -> 3. The plan's
//              codebookByteOffset is the byte offset ADDED to the block base to form the
//              packed-nibble load pointer -- it becomes "3", so plan.codebookByteOffset
//              reaches emit.
//   (2 strip)  a DERIVED geometry field the FormulaProvider COMPUTES (plan.stripLanes =
//              qk/2 for the single-group scale models), which a raw descriptor read does
//              NOT carry: qk 32 -> 44 makes the per-strip vl the codebook nibble load
//              (vle8_v_u8m1) uses become "22" (== 44/2), while the codebook TABLE load
//              (vle8_v_i8m1) keeps "16" (== plan.codebookEntries, independent of qk).
//              Only the provider's qk/2 derivation can produce "22", so this is the
//              definitive witness that the emit consumes the PLAN, not a scattered read.
//
// Byte-exact reproduce-current is proved SEPARATELY (the 4 golden codebook dequant lits
// are unchanged, md5-identical to the pre-phase-4 emit); this file is the load-bearing
// half. The input is the CONSTRUCTED typed region directly (NOT the abstract
// weft_rvv.dequantize_row) so the mutated decode_core attrs are consumed verbatim -- the
// abstract form would re-derive qk/offset from the format name at construction.

module {
  weft.exec.kernel @dequant_mxfp4_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_mxfp4, sew = 32 : i64, source_kernel = "dequant_mxfp4_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "mxfp4", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 17 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "mxfp4", qk = 32 : i64, quant_byte_offset = 1 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 17 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (quant_byte_offset = 1, qk = 32): the plan's codebookByteOffset (1) is the qs
// pointer offset, and the plan's stripLanes (qk/2 = 16) is the per-strip vl the codebook
// nibble load (vle8_v_u8m1) uses; the codebook TABLE load (vle8_v_i8m1) uses
// codebookEntries (16).
// CHECK: emitc.func @weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(
// CHECK: %[[ENTRIES:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK: call_opaque "__riscv_vle8_v_i8m1"(%{{.*}}, %[[ENTRIES]])
// Anchor past the E8M0 scale decode so the qs offset "1" (size_t) is not confused with the
// block-loop step; the FIRST size_t "1" after mxfp4_decode is plan.codebookByteOffset.
// CHECK: verbatim{{.*}}callee=mxfp4_decode
// CHECK: %[[QSOFF:.*]] = literal "1" : !emitc.opaque<"size_t">
// CHECK-NEXT: add %{{.*}}, %[[QSOFF]]
// CHECK: %[[STRIP:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_u8m1
// CHECK-NEXT: call_opaque "__riscv_vle8_v_u8m1"(%{{.*}}, %[[STRIP]])

// MUTQS (quant_byte_offset 1 -> 3): the RE-PACKAGED codebookByteOffset reaches emit -- the
// qs pointer offset becomes "3" (a dead-plan / hard-coded 1 could not fake this away),
// proving the emit CONSUMES plan.codebookByteOffset.
// MUTQS: %[[QSOFF2:.*]] = literal "3" : !emitc.opaque<"size_t">
// MUTQS-NEXT: add %{{.*}}, %[[QSOFF2]]

// MUTSTRIP (qk 32 -> 44): the DERIVED plan.stripLanes = qk/2 = 22 reaches the codebook
// nibble load's vl. A raw descriptor read has no qk/2 field, so "22" can ONLY come through
// the FormulaProvider -- the definitive witness that the emit consumes the PLAN.
// MUTSTRIP: %[[STRIP2:.*]] = literal "22" : !emitc.opaque<"size_t">
// MUTSTRIP-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_u8m1
// MUTSTRIP-NEXT: call_opaque "__riscv_vle8_v_u8m1"(%{{.*}}, %[[STRIP2]])
