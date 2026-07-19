// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
//
// F1 FALSIFIER (phase-1 DequantMechanismPlan nibble slice, C1/C2). Adding a NEW 4-bit
// nibble dequantize_row format (q4_synth) touches ONLY a single descriptor row in
// RVVDequantizeRowConstruction (lookupDequantizeRowStreamFacts) + THIS lit -- ZERO
// emitter / verifier mechanism lines. q4_synth carries q4_0's EXACT decode tuple
// (weight_block_stride 18, quant_byte_offset 2, nibble_bias 8, carrier nibble4, no min /
// no qh), so the constructed typed region dispatches on carrier_kind == "nibble4" (NOT
// the format name) into the SHARED emitDequantizeRowNibbleVectorBody and emits C
// BYTE-IDENTICAL to dequantize_row_q4_0 modulo only the provenance token. This directly
// falsifies "the format name drives the decode dispatch": q4_synth has no per-format
// emitter leaf, no per-format verifier arm, no dispatch if-branch -- only a descriptor
// row. (Name-relative oracle: sed the q4_synth->q4_0 provenance tokens and the emitted C
// is byte-identical to the rvv-to-emitc-ggml-dequantize-row-q4-0 golden.)

module {
  weft.exec.kernel @dequant_q4_synth_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q4_synth attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q4_synth, sew = 32 : i64, source_kernel = "dequant_q4_synth_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q4_synth"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q4_synth_kernel_dequant_q4_synth(
// The construction is real: the emit is DRIVEN by the typed region (the provenance token
// proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS block count nb = k / 32 and the block loop.
// CHECK: div
// CHECK: for
// The fp16 block scale seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The SHARED nibble4 owned real-vector decode (byte-identical to q4_0): the 16 packed
// bytes load, the low/high nibble split (vand_vx / vsrl_vx), the vf4 widen, the -8
// pre-scale bias (nibble_bias 8, READ from the descriptor), the int->float convert, and
// the SINGLE-mul d scale (no min => vfmul_vf, NO fp-contraction). NO gather, NO
// per-format leaf -- the format name q4_synth never keys the dispatch.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
