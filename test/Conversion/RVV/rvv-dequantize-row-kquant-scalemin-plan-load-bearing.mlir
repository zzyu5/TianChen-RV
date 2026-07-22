// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed -e 's/scale_byte_offset = 80 : i64/scale_byte_offset = 90 : i64/' -e 's/kquant_min_byte_offset = 82 : i64/kquant_min_byte_offset = 92 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTSCALE
// RUN: sed 's/quant_byte_offset = 16 : i64/quant_byte_offset = 20 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTQS

// JUDGMENT experiment for phase-4: the KQuantScaleMin MechanismPlan is LOAD-BEARING (the
// K-quant emitter READS plan.*, it is not dead data the emitter re-derives around).
//
// The family-local formula constructs a complete KQuantScaleMinPlan before this
// final typed body exists. This test proves emission consumes its code-affecting
// fields; formula derivation is covered separately by formula behavior tests.
//
//   (1 scale) the RE-PACKAGED scale-block offset AND a DERIVED offset in one shot:
//             scale_byte_offset 80 -> 90. plan.scaleBlockByteOffset becomes "90" (the fp16
//             d read pointer offset), while the final minByteOffset changes 82 -> 92.
//   (2 qsoff) a RE-PACKAGED geometry field: quant_byte_offset 16 -> 20. The plan's
//             quantByteOffset is the byte offset ADDED to the block base to form the qs
//             (packed-2bit) load pointer for each sub-group -- it becomes "20", a literal
//             that does NOT appear in the unmutated emit, so a dead-plan / hard-coded 16
//             could not fake this: plan.quantByteOffset reaches emit.
//
// Byte-exact reproduce-current is proved SEPARATELY (the 5 golden K-quant dequant lits are
// unchanged, md5-identical to the pre-phase-4 emit); this file is the load-bearing half.
// The input is the CONSTRUCTED typed region directly (NOT the abstract
// weft_rvv.dequantize_row) so the mutated decode_core attrs are consumed verbatim -- the
// abstract form would re-derive qk/offset from the format name at construction.

module {
  weft.exec.kernel @dequant_q2_k_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q2_k attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q2_K", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 84 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "q2_K", dequant_load_lmul = "m1", dequant_mechanism = "kquant-scale-min", dequant_strip_lanes = 16 : i64, kquant_has_high_bit_plane = false, kquant_has_min = true, kquant_high_bit_byte_offset = 0 : i64, kquant_min_byte_offset = 82 : i64, kquant_scale_model = "q2k", kquant_sub_scale_byte_offset = 0 : i64, qk = 256 : i64, quant_byte_offset = 16 : i64, scale_byte_offset = 80 : i64, weight_block_stride = 84 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (scale_byte_offset = 80, quant_byte_offset = 16): the plan's scaleBlockByteOffset
// (80) is the fp16 d read offset, the DERIVED minByteOffset (80 + 2 = 82) is the fp16 dmin
// read offset, and quantByteOffset (16) is the first sub-group's qs load offset.
// CHECK: emitc.func @weft_emitc_dequant_q2_k_kernel_dequant_q2_k(
// CHECK: verbatim{{.*}}callee=q2_K_decode
// CHECK: %[[DOFF:.*]] = literal "80" : !emitc.opaque<"size_t">
// CHECK-NEXT: add %{{.*}}, %[[DOFF]]
// CHECK: %[[MOFF:.*]] = literal "82" : !emitc.opaque<"size_t">
// CHECK-NEXT: add %{{.*}}, %[[MOFF]]
// CHECK: %[[QSOFF:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK-NEXT: add %{{.*}}, %[[QSOFF]]
// CHECK: call_opaque "__riscv_vle8_v_u8m1"

// MUTSCALE (scale_byte_offset 80 -> 90): the RE-PACKAGED scaleBlockByteOffset reaches emit
// (d read offset "90"), and the final minByteOffset 92 reaches the dmin read.
// MUTSCALE: verbatim{{.*}}callee=q2_K_decode
// MUTSCALE: %[[DOFF2:.*]] = literal "90" : !emitc.opaque<"size_t">
// MUTSCALE-NEXT: add %{{.*}}, %[[DOFF2]]
// MUTSCALE: %[[MOFF2:.*]] = literal "92" : !emitc.opaque<"size_t">
// MUTSCALE-NEXT: add %{{.*}}, %[[MOFF2]]

// MUTQS (quant_byte_offset 16 -> 20): the RE-PACKAGED quantByteOffset reaches emit -- the qs
// (packed-2bit) load pointer offset becomes "20" (a literal absent from the unmutated emit,
// so a hard-coded 16 could not fake this away), proving the emit CONSUMES plan.quantByteOffset.
// MUTQS: verbatim{{.*}}callee=q2_K_decode
// MUTQS: %[[QSOFF2:.*]] = literal "20" : !emitc.opaque<"size_t">
// MUTQS-NEXT: add %{{.*}}, %[[QSOFF2]]
