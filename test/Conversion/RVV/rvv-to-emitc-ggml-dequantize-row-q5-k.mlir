// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q5_K` super-block DECODE (block_q5_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q5_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q5_K"); typed_dequantize_row_loop_yield }
// and lowers it via the OWNED REAL-VECTOR body emitDequantizeRowQ45KVectorBody (R线 §四.2,
// isQ5 gate): the emission is DRIVEN by the typed region op-identity + decode_model
// ([L-6]/[L-8] construction, NOT the abstract format string). Byte-exact-vs-ggml-reference
// dequantize_row_q5_K by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v; the dispatch-wired monolith keeps the scalar
// emitDequantizeRowKQuantBodyShared): the fp16 d/dmin seam, the get_scale_min_k4 6-bit unpack
// in SCALAR C, then the per-super-sub 32-lane nibble pipeline plus the per-lane qh 5th-bit
// merge folded d1*q - m1 in ONE fused vfmsac (matching the -ffp-contract=on opponent). NO
// gather. ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for q5_K dequant.

module {
  weft.exec.kernel @dequant_q5_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q5_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q5_K, sew = 32 : i64, source_kernel = "dequant_q5_K_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q5_K"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q5_K_kernel_dequant_q5_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR nibble + 5th-bit decode (R线 §四.2 K-quant fan-out): the 32
// packed nibble bytes + the 32 qh bytes load, the nibble split (vand_vx), the per-lane
// 5th-bit spread (vsll_vx/vor_vv from qh), the vf4 widen + convert, and the FUSED
// d1*v - m1 fold (vfmv_v_f(m1) + vfmsac_vf(d1), matching the contracted opponent). NO
// gather. OWNED __riscv_v (ISSUE-001 reverse; closes ISSUE-002 for q5_K dequant).
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: bitwise_and
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vsll_vx_u8m2"
// CHECK: call_opaque "__riscv_vor_vv_u8m2"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmsac_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
