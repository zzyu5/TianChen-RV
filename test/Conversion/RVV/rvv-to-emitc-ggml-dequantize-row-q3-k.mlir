// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q3_K` super-block DECODE (block_q3_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q3_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q3_K"); typed_dequantize_row_loop_yield }
// and lowers it via the OWNED REAL-VECTOR body emitDequantizeRowQ3KVectorBody (R线 §四.2):
// the emission is DRIVEN by the typed region op-identity + decode_model ([L-6]/[L-8]
// construction, NOT the abstract format string). Byte-exact-vs-ggml-reference
// dequantize_row_q3_K by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v; the monolith keeps the scalar
// emitDequantizeRowKQuantBodyShared): the fp16 d_all seam, the 6-bit signed scales from the
// packed scales\[12\] (aux kmask1/kmask2 shuffle in SCALAR C, byte-exact), then the per-group
// 16-lane 2-bit quant merged with the hmask high-bit term folded dl*qdec in a SINGLE mul
// (NO min -> no fp-contraction). NO gather. ISSUE-001 reverse; closes ISSUE-002 for q3_K.

module {
  weft.exec.kernel @dequant_q3_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q3_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q3_K"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q3_K_kernel_dequant_q3_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The 6-bit aux kmask scale shuffle (SCALAR C, byte-exact to ggml).
// CHECK: bitwise_and
// The OWNED REAL-VECTOR 2-bit + hmask-high-bit decode (R线 §四.2 K-quant fan-out): the
// 16 qs bytes + 16 hmask bytes load, the 2-bit extract (vsrl_vx + vand_vx), the hmask
// high term ((hm>>mbit)&1 -> (1-bit)<<2 via vrsub_vx) in i32, qdec = qbits - term, the
// convert, and the SINGLE-mul dl fold (vfmul_vf, NO min -> no fp-contraction). NO
// gather. OWNED __riscv_v (ISSUE-001 reverse; closes ISSUE-002 for q3_K dequant).
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vrsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
