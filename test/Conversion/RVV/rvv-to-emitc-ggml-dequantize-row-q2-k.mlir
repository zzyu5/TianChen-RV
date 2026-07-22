// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q2_K` super-block DECODE (block_q2_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q2_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q2_K"); typed_dequantize_row_loop_yield }
// and lowers it via the OWNED REAL-VECTOR body emitDequantizeRowQ2KVectorBody (R线 §四.2):
// the emission is DRIVEN by the typed region op-identity + decode_model ([L-6]/[L-8]
// construction, NOT the abstract format string). Byte-exact-vs-ggml-reference
// dequantize_row_q2_K by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v; the monolith keeps the scalar
// emitDequantizeRowKQuantBodyShared): the fp16 d/dmin seam, the 4-bit packed scale/min
// (sc&0xF, sc>>4) in SCALAR C, then the per-group 16-lane 2-bit quant pipeline ((q>>shift)&3)
// folded dl*q - ml in ONE fused vfmsac (matching the -ffp-contract=on opponent). NO gather.
// ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for q2_K dequant.

module {
  weft.exec.kernel @dequant_q2_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q2_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q2_K"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q2_K_kernel_dequant_q2_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body,
// not the dispatch-wired monolith).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS super-block count nb = k / 256 and the block loop.
// CHECK: div
// CHECK: for
// The two fp16 super-block scales (d, dmin) through the shared seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The 4-bit scale/min unpack (SCALAR C, byte-exact to ggml).
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
// The OWNED REAL-VECTOR 2-bit quant decode (R线 §四.2 K-quant fan-out): the 16 quant
// bytes load, the 2-bit extract (vsrl_vx(shift) + vand_vx(3)), the vf4 widen + convert,
// and the FUSED dl*q - ml fold (vfmv_v_f(ml) + vfmsac_vf(dl), matching the contracted
// opponent). NO gather. OWNED __riscv_v (ISSUE-001 reverse; closes ISSUE-002 for q2_K).
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmsac_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
