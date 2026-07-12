// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq2_s` block DECODE (block_iq2_s -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the IQ grid-table dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq2_s") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq2_s", qk=256, stride=82);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). The emitted C is BYTE-IDENTICAL to the dispatch-wired iq2_s monolith modulo
// ONLY the source-op provenance token (the SHARED grid decode emitGgmlDequantizeRowExtended,
// reached via emitDequantizeRowIQGridBodyShared, is the SAME code both paths run) -- the
// 1024-entry (int64) grid (qs | qh-derived high bits) + the universal signs256 sign plane
// indexed by the EXPLICIT sign byte, the per-ib32 4-bit scales d*(0.5+scale)*0.25, and the
// fp16 d via the (float)*(const _Float16 *) seam. The grid + signs256 tables are DERIVED at
// emit as function-local statics (NO signs op-attr; that blocker is block-dot-repack-only,
// not this streaming dequant path). REUSES the iq2_s block-dot vec_dot grid decl. Byte-exact-
// vs-ggml-reference dequantize_row_iq2_s (a scalar AoS super-block loop; no reduction).

module {
  weft.exec.kernel @dequant_iq2_s_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq2_s attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq2_s, sew = 32 : i64, source_kernel = "dequant_iq2_s_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq2_s"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq2_s_kernel_dequant_iq2_s(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The 1024-entry (int64) grid decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq2s_grid
// The universal signs256 sign plane decl, emitted once above the super-block loop.
// CHECK: weft_iq2s_signs256
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
