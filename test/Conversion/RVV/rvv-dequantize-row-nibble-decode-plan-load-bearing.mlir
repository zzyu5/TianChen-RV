// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/nibble_bias = 8 : i64/nibble_bias = 5 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTBIAS
// RUN: sed 's/qk = 32 : i64/qk = 44 : i64/g' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTSTRIP

// JUDGMENT experiment for the typed NibbleDecode decision: the MechanismPlan is
// LOAD-BEARING (the
// nibble emitter READS plan.*, it is not dead data the emitter scatter-reads around).
//
// Phase-2 introduced weft::NibbleDecodePlan (Support) + the FormulaProvider
// decideNibbleDecode (RVVFormulaDecision.h) and wired the flat-nibble emit so
// the ONLY path from the decode_core descriptor to the emitted C is
//   decode_core typed g -> decideNibbleDecode -> NibbleDecodePlan -> emit.
// The per-attr scatter reads (getNibbleBiasAttr / getMinByteOffsetAttr / ... straight
// into the shared body) are RETIRED. This test proves the plan is really consumed by
// mutating two DIFFERENT kinds of plan field and observing the emit change:
//
//   (1 bias)   a RE-PACKAGED 8-tuple field: nibble_bias 8 -> 5. The pre-scale bias the
//              vsub_vx_i32m4 subtracts becomes "5" -- so plan.nibbleBias reaches emit.
//   (2 strip)  a DERIVED geometry field the FormulaProvider COMPUTES (plan.stripLanes =
//              qk/2), which a raw descriptor read does NOT carry: qk 32 -> 44 makes the
//              per-strip vl the vle8 nibble load uses become "22" (== 44/2). Only the
//              provider's qk/2 derivation can produce "22", so this is the definitive
//              witness that the emit consumes the PLAN, not a scattered descriptor read.
//
// Byte-exact reproduce-current is proved SEPARATELY (the 5 golden dequant lits are
// unchanged, md5-identical to the post-phase-1 emit); this file is the load-bearing half.

module {
  weft.exec.kernel @dequant_q4_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q4_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q4_0, sew = 32 : i64, source_kernel = "dequant_q4_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q4_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 18 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {carrier_kind = "nibble4", decode_model = "q4_0", nibble_bias = 8 : i64, qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 18 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (nibble_bias = 8, qk = 32): the plan's stripLanes (qk/2 = 16) is the per-strip
// vl the u8 nibble load uses, and the plan's nibbleBias (8) is the pre-scale vsub bias.
// CHECK: emitc.func @weft_emitc_dequant_q4_0_kernel_dequant_q4_0(
// CHECK: %[[VL:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK: call_opaque "__riscv_vle8_v_u8m1"(%{{.*}}, %[[VL]])
// CHECK: %[[BIAS:.*]] = literal "8" : !emitc.opaque<"int32_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vsub_vx_i32m4
// CHECK-NEXT: call_opaque "__riscv_vsub_vx_i32m4"(%{{.*}}, %[[BIAS]], %[[VL]])

// MUTBIAS (nibble_bias 8 -> 5): the RE-PACKAGED 8-tuple field reaches emit -- the vsub
// bias becomes "5" (falsify-proof byte-diff; a dead-plan / hard-coded 8 could not fake
// this away), proving the emit CONSUMES plan.nibbleBias.
// MUTBIAS: %[[VL2:.*]] = literal "16" : !emitc.opaque<"size_t">
// MUTBIAS: %[[BIAS2:.*]] = literal "5" : !emitc.opaque<"int32_t">
// MUTBIAS-NEXT: verbatim{{.*}}callee=__riscv_vsub_vx_i32m4
// MUTBIAS-NEXT: call_opaque "__riscv_vsub_vx_i32m4"(%{{.*}}, %[[BIAS2]], %[[VL2]])

// MUTSTRIP (qk 32 -> 44): the DERIVED plan.stripLanes = qk/2 = 22 reaches the u8 nibble
// load's vl. A raw descriptor read has no qk/2 field, so "22" can ONLY come through the
// FormulaProvider -- the definitive witness that the emit consumes the PLAN.
// MUTSTRIP: %[[VL3:.*]] = literal "22" : !emitc.opaque<"size_t">
// MUTSTRIP: call_opaque "__riscv_vle8_v_u8m1"(%{{.*}}, %[[VL3]])
