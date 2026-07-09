// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q6_K` super-block DECODE (block_q6_K -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the K-quant dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract tcrv_rvv.dequantize_row
// (format="q6_K") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed tcrv_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q6_K"); typed_dequantize_row_loop_yield }
// and lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q6_K monolith modulo ONLY the source-op
// provenance token (the SHARED super-block decode emitGgmlDequantizeRowExtended, reached
// via emitDequantizeRowKQuantBodyShared, is the SAME code both paths run) -- the fp16 d
// scale via the (float)*(const _Float16 *) seam, the SIGNED int8 scales\[16\] loads, and
// the 6-bit quant assembly ((ql&0xF) | ((qh>>s)&3)<<4) - 32 folded d*sc*q. REUSES the
// q6_K block-dot vec_dot decode. Byte-exact-vs-ggml-reference dequantize_row_q6_K (a
// scalar AoS super-block loop; no reduction).

module {
  tcrv.exec.kernel @dequant_q6_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q6_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q6_K, sew = 32 : i64, source_kernel = "dequant_q6_K_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q6_K"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q6_K_kernel_dequant_q6_K(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The SIGNED int8 super-block scales (sign-extending load).
// CHECK: const int8_t
// The 6-bit quant assembly (ql nibble | qh high 2 bits).
// CHECK: bitwise_and
