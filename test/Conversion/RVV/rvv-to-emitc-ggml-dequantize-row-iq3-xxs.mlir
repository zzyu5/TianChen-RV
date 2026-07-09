// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq3_xxs` block DECODE (block_iq3_xxs -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the IQ grid-table dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract tcrv_rvv.dequantize_row
// (format="iq3_xxs") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed tcrv_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq3_xxs", qk=256, stride=98);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). The emitted C is BYTE-IDENTICAL to the dispatch-wired iq3_xxs monolith modulo
// ONLY the source-op provenance token (the SHARED grid decode emitGgmlDequantizeRowExtended,
// reached via emitDequantizeRowIQGridBodyShared, is the SAME code both paths run) -- the
// grid-of-4 (uint32) codebook + the ksigns selector plane (per-lane bit-tested), the
// per-ib32 aux scale d*(0.5+(aux>>28))*0.5, and the fp16 d via the (float)*(const _Float16 *)
// seam. The grid + ksigns tables are DERIVED at emit as function-local statics (NO ksigns
// op-attr; that blocker is block-dot-repack-only, not this streaming dequant path). REUSES
// the iq3_xxs block-dot vec_dot grid decl. Byte-exact-vs-ggml-reference dequantize_row_iq3_xxs
// (a scalar AoS super-block loop; no reduction).

module {
  tcrv.exec.kernel @dequant_iq3_xxs_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_iq3_xxs attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq3_xxs, sew = 32 : i64, source_kernel = "dequant_iq3_xxs_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "iq3_xxs"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_iq3_xxs_kernel_dequant_iq3_xxs(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// The grid-of-4 (uint32) codebook decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: tcrv_iq3xxs_grid
// The ksigns selector plane decl, emitted once above the super-block loop.
// CHECK: tcrv_iq3xxs_ksigns
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
