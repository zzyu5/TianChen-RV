// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_tq1_0` super-block DECODE (block_tq1_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the ternary super-block dequant leaf (a dequantize_row
// family member, family-head q8_0): the abstract tcrv_rvv.dequantize_row (format="tq1_0")
// is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites it into the typed
// tcrv_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core (decode_model
// "tq1_0", qk=256, stride=54); typed_dequantize_row_loop_yield } and lowers it (the emission
// is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8] construction, NOT the
// abstract format string). The emitted C is BYTE-IDENTICAL to the dispatch-wired tq1_0 monolith
// modulo ONLY the source-op provenance token (the SHARED decode emitGgmlDequantizeRowExtended,
// reached via emitDequantizeRowCodebookGridBodyShared, is the SAME code both paths run) --
// the fp16 d scale via the (float)*(const _Float16 *) seam (@52, block END), then the base-3
// unpack q = (qs*pow3[n]) mod 256, xi = ((q*3)>>8), y = (xi-1)*d over the qs (160+80) and qh
// (16) sections. The pow3 weight table is DERIVED at emit as a function-local static (NO
// op-attr). REUSES the tq1_0 block-dot vec_dot decode. Byte-exact-vs-ggml-reference
// dequantize_row_tq1_0 (a scalar AoS super-block loop; no reduction).

module {
  tcrv.exec.kernel @dequant_tq1_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_tq1_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_tq1_0, sew = 32 : i64, source_kernel = "dequant_tq1_0_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "tq1_0"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_tq1_0_kernel_dequant_tq1_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH tcrv_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// The pow3 base-3 weight table decl, emitted once above the loop.
// CHECK: static const uint8_t tcrv_dequant_tq1_0_pow3
// CHECK: div
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The base-3 unpack: (q*3)>>8 with the uint8 truncation & 0xFF.
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
