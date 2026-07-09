// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q8_0` block DECODE (block_q8_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the dequant FRONT DOOR (family-head of the
// 23-format spectrum, G3 line-B): the abstract tcrv_rvv.dequantize_row (format="q8_0")
// is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites it into the
// typed tcrv_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the
// typed region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract
// format string). The emitted C is BYTE-IDENTICAL to the retired dispatch-wired q8_0
// monolith modulo ONLY the source-op provenance token (the shared body emitter
// emitDequantizeRowQ8_0BodyShared is common to both) -- the fp16 scale via the
// (float)*(const _Float16 *) seam, then the bare signed-int8 scale y[j] = qs[j]*d over
// all 32 block lanes (the load is a `const int8_t` read that sign-extends). The other
// 22 dequantize_row formats stay dispatch-wired (hand-written per-format monolith).
// The dedicated typed-region -> C lowering contract (+ verifier fail-closed) is locked
// in rvv-to-emitc-typed-dequantize-row-loop-body.mlir. Byte-exact-vs-ggml-reference
// dequantize_row_q8_0 (a scalar AoS block loop; no reduction).

module {
  tcrv.exec.kernel @dequant_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q8_0, sew = 32 : i64, source_kernel = "dequant_q8_0_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q8_0"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q8_0_kernel_dequant_q8_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body,
// not the dispatch-wired monolith).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// The AoS block count + the block loop.
// CHECK: div
// CHECK: for
// The fp16 block scale seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The signed-int8 read (sign-extending) + the bare block scale.
// CHECK: !emitc.opaque<"const int8_t">
// CHECK: mul
