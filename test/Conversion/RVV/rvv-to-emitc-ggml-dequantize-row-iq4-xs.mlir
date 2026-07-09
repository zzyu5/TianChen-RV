// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq4_xs` block DECODE (block_iq4_xs -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the super-block non-linear codebook dequant leaf (G3
// line-B, a dequantize_row family member, family-head q8_0): the abstract
// tcrv_rvv.dequantize_row (format="iq4_xs") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// tcrv_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "iq4_xs", qk=256, stride=136); typed_dequantize_row_loop_yield } and lowers
// it (the emission is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8]
// construction, NOT the abstract format string). The emitted C is BYTE-IDENTICAL to the
// dispatch-wired iq4_xs monolith modulo ONLY the source-op provenance token (the SHARED
// decode emitGgmlDequantizeRowExtended, reached via emitDequantizeRowCodebookGridBodyShared,
// is the SAME code both paths run) -- the fp16 d via the (float)*(const _Float16 *) seam,
// the 6-bit ls from a scales_l nibble + a scales_h 2-bit (dl=d*(ls-32)), then the 16-entry
// iq4_nl non-linear codebook nibble decode. The codebook is DERIVED at emit as a
// function-local static (NO codebook op-attr; that blocker is block-dot-repack-only, not
// this streaming dequant path). REUSES the iq4_nl block-dot vec_dot codebook.
// Byte-exact-vs-ggml-reference dequantize_row_iq4_xs (a scalar AoS super-block loop; no
// reduction).

module {
  tcrv.exec.kernel @dequant_iq4_xs_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_iq4_xs attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq4_xs, sew = 32 : i64, source_kernel = "dequant_iq4_xs_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "iq4_xs"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_iq4_xs_kernel_dequant_iq4_xs(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// The codebook table decl, emitted once above the super-block loop (reused from the vec_dot codebook).
// CHECK: tcrv_dequant_iq4nl_kvalues
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for %
// CHECK: call_opaque "(float)*(const _Float16 *)"
