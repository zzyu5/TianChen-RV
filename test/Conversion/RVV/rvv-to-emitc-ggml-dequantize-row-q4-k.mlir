// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q4_K` super-block DECODE (block_q4_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q4_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q4_K"); typed_dequantize_row_loop_yield }
// and lowers it via the OWNED REAL-VECTOR body emitDequantizeRowQ45KVectorBody (R线 §四.2
// K-quant fan-out): the emission is DRIVEN by the typed region op-identity + decode_model
// ([L-6]/[L-8] construction, NOT the abstract format string). Byte-exact-vs-ggml-reference
// dequantize_row_q4_K by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith keeps
// the scalar emitDequantizeRowKQuantBodyShared): the fp16 d/dmin scales via the
// (float)*(const _Float16 *) seam, the get_scale_min_k4 6-bit scale/min unpack in SCALAR C,
// then the per-super-sub 32-lane nibble pipeline (q&0xF, q>>4) folded d1*q - m1 in ONE fused
// vfmsac (matching the -ffp-contract=on opponent). NO gather (K-quant is bit-unpack, not a
// codebook lookup). ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for q4_K dequant.

module {
  weft.exec.kernel @dequant_q4_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q4_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q4_K, sew = 32 : i64, source_kernel = "dequant_q4_K_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q4_K"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q4_K_kernel_dequant_q4_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// CHECK: for
// The two fp16 super-block scales (d, dmin).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The get_scale_min_k4 6-bit scale/min unpack (SCALAR C, byte-exact to ggml).
// CHECK: bitwise_and
// The OWNED REAL-VECTOR nibble decode (R线 §四.2 K-quant fan-out, PR-31 precedent): the
// 32 packed bytes load, the low nibble split (vand_vx), the vf4 widen, the int->float
// convert, and the FUSED d1*v - m1 fold (vfmv_v_f(m1) + vfmsac_vf(d1), ONE rounding
// matching the -ffp-contract=on opponent's vfmsub), then the high nibble split
// (vsrl_vx). NO gather. The vector content is the EMITTER's (OWNED __riscv_v), not
// host-autovec lottery (the ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery
// exposure for q4_K dequant).
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m8"
// CHECK: call_opaque "__riscv_vreinterpret_v_u32m8_i32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m8"
// CHECK: call_opaque "__riscv_vfmsac_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m2"
