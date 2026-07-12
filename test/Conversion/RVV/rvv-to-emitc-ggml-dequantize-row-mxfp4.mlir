// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_mxfp4` block DECODE (block_mxfp4 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the FP4 codebook dequant leaf (G3 line-B negative
// control, a dequantize_row family member, family-head q8_0): the abstract
// weft_rvv.dequantize_row (format="mxfp4") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "mxfp4", qk=32, stride=17); typed_dequantize_row_loop_yield } and lowers it
// (the emission is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8]
// construction, NOT the abstract format string). The emitted C is BYTE-IDENTICAL to the
// dispatch-wired mxfp4 monolith modulo ONLY the source-op provenance token (the SHARED
// decode emitGgmlDequantizeRowExtended, reached via emitDequantizeRowCodebookGridBodyShared,
// is the SAME code both paths run) -- the E8M0 block scale reconstructed by ggml's exact
// bit construction (0x00200000u denormal / (e-1)<<23 normal), then the 16-entry FP4 (e2m1)
// codebook gather (kvalues_mxfp4[qs&0xF], kvalues_mxfp4[qs>>4]) scaled by d. The codebook +
// E8M0 scale are DERIVED at emit (NO op-attr; that blocker is block-dot-repack-only, not
// this streaming dequant path). REUSES the mxfp4 block-dot vec_dot codebook + E8M0 scale
// decode. Byte-exact-vs-ggml-reference dequantize_row_mxfp4 (a scalar AoS block loop; no
// reduction).

module {
  weft.exec.kernel @dequant_mxfp4_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_mxfp4, sew = 32 : i64, source_kernel = "dequant_mxfp4_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "mxfp4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The FP4 codebook table decl (kvalues_mxfp4), emitted once above the loop.
// CHECK: static const int8_t weft_dequant_mxfp4_kvalues
// The block count nb = k / 32 and the block loop.
// CHECK: div
// CHECK: for
// The E8M0 -> fp32 HALF scale bit construction (NO fp16 seam).
// CHECK: 0x00200000u
// CHECK: bitwise_and
