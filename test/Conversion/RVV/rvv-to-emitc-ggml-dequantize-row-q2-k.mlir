// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q2_K` super-block DECODE (block_q2_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q2_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q2_K"); typed_dequantize_row_loop_yield }
// and lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q2_K monolith modulo ONLY the source-op
// provenance token (the SHARED super-block decode emitGgmlDequantizeRowExtended, reached
// via emitDequantizeRowKQuantBodyShared, is the SAME code both paths run) -- the fp16
// d/dmin scales via the (float)*(const _Float16 *) seam, then the 4-bit packed scale/min
// (sc&0xF, sc>>4) and the 2-bit quant unpack ((q>>shift)&3) folded dl*q - ml. The q2_K
// decode REUSES the block-decode facts already built for the q2_K block-dot vec_dot.
// Byte-exact-vs-ggml-reference dequantize_row_q2_K (a scalar AoS super-block loop; no
// reduction).

module {
  weft.exec.kernel @dequant_q2_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q2_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q2_K, sew = 32 : i64, source_kernel = "dequant_q2_K_kernel", status = "selected-lowering-boundary"} {
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
// The 4-bit scale/min unpack and the 2-bit quant shift.
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
