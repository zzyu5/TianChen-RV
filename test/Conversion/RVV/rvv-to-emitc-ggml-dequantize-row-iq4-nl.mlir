// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq4_nl` block DECODE (block_iq4_nl -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the non-linear codebook dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq4_nl") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq4_nl", qk=32, stride=18);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). The emitted C is BYTE-IDENTICAL to the dispatch-wired iq4_nl monolith modulo
// ONLY the source-op provenance token (the SHARED decode emitGgmlDequantizeRowExtended,
// reached via emitDequantizeRowCodebookGridBodyShared, is the SAME code both paths run) --
// the fp16 d via the (float)*(const _Float16 *) seam, then the 16-entry non-linear codebook
// gather (kvalues_iq4nl[qs&0xF], kvalues_iq4nl[qs>>4]) scaled by d. The codebook is DERIVED
// at emit as a function-local static (NO codebook op-attr; that blocker is
// block-dot-repack-only, not this streaming dequant path). REUSES the iq4_nl block-dot
// vec_dot codebook. Byte-exact-vs-ggml-reference dequantize_row_iq4_nl (a scalar AoS block
// loop; no reduction).

module {
  weft.exec.kernel @dequant_iq4_nl_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq4_nl attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq4_nl, sew = 32 : i64, source_kernel = "dequant_iq4_nl_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq4_nl"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq4_nl_kernel_dequant_iq4_nl(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The 16-entry non-linear codebook table decl, emitted once above the loop.
// CHECK: static const int8_t weft_dequant_iq4nl_kvalues
// CHECK: div
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The nibble split feeding the codebook gather.
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
