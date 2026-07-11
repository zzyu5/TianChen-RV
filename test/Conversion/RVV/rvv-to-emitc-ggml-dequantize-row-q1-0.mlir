// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q1_0` block DECODE (block_q1_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the flat 1-bit binary-sign dequant leaf (a
// dequantize_row family member, family-head q8_0): the abstract
// tcrv_rvv.dequantize_row (format="q1_0") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// tcrv_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "q1_0", qk=128, stride=18); typed_dequantize_row_loop_yield } and
// lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q1_0 monolith fallback modulo ONLY the
// source-op provenance token (the SHARED decode emitGgmlDequantizeRowExtended is the
// SAME code both paths run) -- the fp16 d scale via the (float)*(const _Float16 *)
// seam (@0), neg_d = -d, then the packed 1-bit sign unpack bit = (qs[j/8] >> (j%8))
// & 1, y[j] = bit ? d : neg_d over the 16 qs bytes (@2) x 8 bits = 128 lanes.
// Byte-exact-vs-ggml-reference dequantize_row_q1_0 (a scalar AoS block loop; no
// reduction) -- host random-block verification GREEN (0 mismatches).

module {
  tcrv.exec.kernel @dequant_q1_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q1_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q1_0, sew = 32 : i64, source_kernel = "dequant_q1_0_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q1_0"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q1_0_kernel_dequant_q1_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// The AoS block count nb = k / 128 + the block loop.
// CHECK: div
// CHECK: for
// The fp16 d scale seam + neg_d = -d.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: unary_minus
// The 1-bit sign unpack: (qs[j/8] >> (j%8)) & 1, then bit ? d : neg_d.
// CHECK: bitwise_right_shift
// CHECK: bitwise_and
// CHECK: conditional
