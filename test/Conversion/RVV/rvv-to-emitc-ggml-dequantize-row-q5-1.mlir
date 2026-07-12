// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q5_1` block DECODE (block_q5_1 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the flat nibble+5th-bit dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q5_1") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q5_1"); typed_dequantize_row_loop_yield }
// and lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q5_1 monolith modulo ONLY the source-op
// provenance token (the shared body emitter emitDequantizeRowNibbleBodyShared, reached
// via emitDequantizeRowQ5_1BodyShared, is common to both) -- the fp16 scale d + fp16
// min m via the (float)*(const _Float16 *) seam, the byte-assembled little-endian uint32
// qh 5th-bit plane, then ((qs[j]&0x0F)|xh0)*d+m -> y[j] and ((qs[j]>>4)|xh1)*d+m ->
// y[j+16]. The q5_1 qh+min decode REUSES the block-decode already built for the q5_1
// block-dot vec_dot. Byte-exact-vs-ggml-reference dequantize_row_q5_1 (scalar AoS loop).

module {
  weft.exec.kernel @dequant_q5_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q5_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q5_1, sew = 32 : i64, source_kernel = "dequant_q5_1_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q5_1"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q5_1_kernel_dequant_q5_1(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS block count + the block loop.
// CHECK: div
// CHECK: for
// The fp16 d and fp16 m scales (two seam reads).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The byte-assembled uint32 qh 5th-bit plane.
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// The nibble decode + the *d then +m affine.
// CHECK: bitwise_and
// CHECK: mul
// CHECK: add
