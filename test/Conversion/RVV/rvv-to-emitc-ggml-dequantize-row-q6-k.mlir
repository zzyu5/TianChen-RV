// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q6_K` super-block DECODE (block_q6_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q6_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q6_K"); typed_dequantize_row_loop_yield }
// and lowers it via the OWNED REAL-VECTOR body emitDequantizeRowQ6KVectorBody (R线 §四.2):
// the emission is DRIVEN by the typed region op-identity + decode_model ([L-6]/[L-8]
// construction, NOT the abstract format string). Byte-exact-vs-ggml-reference
// dequantize_row_q6_K by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v; the monolith keeps the scalar
// emitDequantizeRowKQuantBodyShared): the fp16 d seam, the SIGNED int8 scales\[16\] loads,
// then the per-group 16-lane 6-bit quant assembly ((ql&0xF) | ((qh>>s)&3)<<4) - 32 folded
// (d*sc)*q in a SINGLE mul (NO add -> no fp-contraction; the l/16 scale split -> two 16-lane
// groups per 32-span). NO gather. ISSUE-001 reverse; closes ISSUE-002 for q6_K dequant.

module {
  weft.exec.kernel @dequant_q6_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q6_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q6_K, sew = 32 : i64, source_kernel = "dequant_q6_K_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q6_K"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q6_K_kernel_dequant_q6_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The SIGNED int8 super-block scales (sign-extending load).
// CHECK: const int8_t
// The OWNED REAL-VECTOR 6-bit quant decode (R线 §四.2 K-quant fan-out): the 16 ql bytes
// + 16 qh bytes load, the nibble (vand_vx / vsrl_vx) | the qh high 2 bits (vsll_vx +
// vor_vv) -> combined 0..63, the vf4 widen, the -32 bias (vsub_vx), the convert, and the
// SINGLE-mul dsc fold (vfmul_vf, `(d*sc)*q`, NO fp-contraction). NO gather. OWNED
// __riscv_v (ISSUE-001 reverse; closes ISSUE-002 for q6_K dequant).
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsll_vx_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
