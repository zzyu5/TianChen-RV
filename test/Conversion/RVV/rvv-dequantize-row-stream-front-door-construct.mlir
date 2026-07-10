// RUN: tcrv-opt %s --tcrv-rvv-materialize-dequantize-row-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-dequantize-row-stream-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's/"q8_0"/"tq2_0"/g' %s | tcrv-opt --tcrv-rvv-materialize-dequantize-row-stream-front-door | FileCheck %s --check-prefix=TERNARY

// CERT-FD首族 (dequant×21) -- the PRE-EMITC dequant-stream FRONT DOOR. It runs ONLY
// the CONSTRUCTION half of constructOrEmitGgmlDequantizeRow (the shared byte-exact
// tcrv::rvv::constructTypedDequantizeRowLoopBody): it rewrites the abstract
// tcrv_rvv.dequantize_row into the typed tcrv_rvv.typed_dequantize_row_loop_body region
//   { dequantize_row_decode_core; typed_dequantize_row_loop_yield }
// and STOPS -- BEFORE --tcrv-rvv-lower-to-emitc. This exposes the constructed region so
// the certification walker (e5_strong_readout.py stamp-dequant-stream) can WALK (and hence
// machine-certify) the realized typed region, instead of it being built-and-erased atomically
// inside the emitc lowering where no pre-emitc dump can see it.
//
// The construction is the SAME one the in-emitc fallback runs, so the emitted C is byte-exact
// whether the region is built here (pre-emitc, REALIZE) or in emitc: the EMIT run below (front
// door THEN --tcrv-rvv-lower-to-emitc) is byte-identical to the atomic
// `--tcrv-rvv-lower-to-emitc`-only path locked by rvv-to-emitc-ggml-dequantize-row-q8-0.mlir
// (the 0-diff is verified format-by-format across all 23 constructed formats out-of-band).
// The ternary super-blocks (tq1_0/tq2_0) are NOW front-door CONSTRUCTED too: the front door
// rewrites their abstract op into the SAME typed region (TERNARY run below), completing the
// whole 23-format dequantize_row spectrum -- NO format remains dispatch-wired. Numerical
// semantics: zero change (byte-exact by construction).

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

// The front door CONSTRUCTS the typed streaming region and STOPS pre-emitc: the abstract
// tcrv_rvv.dequantize_row is GONE, replaced by the typed body carrying the decode core + yield.
// REALIZE-NOT: tcrv_rvv.dequantize_row {{[^_]}}
// REALIZE-NOT: emitc.
// REALIZE: tcrv_rvv.typed_dequantize_row_loop_body %{{.*}}, %{{.*}}, %{{.*}} attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64}
// REALIZE: ^bb0(%[[BI:.*]]: index):
// REALIZE: tcrv_rvv.dequantize_row_decode_core %{{.*}}, %{{.*}}, %[[BI]] {decode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64}
// REALIZE: tcrv_rvv.typed_dequantize_row_loop_yield

// Front door THEN emitc == the atomic construct+emit path: the emit is DRIVEN by the typed
// region (the provenance token proves the abstract op went THROUGH the typed region), and the
// C is byte-identical to the atomic q8_0 path.
// EMIT-NOT: tcrv_rvv.
// EMIT: emitc.func @tcrv_emitc_dequant_q8_0_kernel_dequant_q8_0(
// EMIT: route_source_op=tcrv_rvv.typed_dequantize_row_loop_body
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: !emitc.opaque<"const int8_t">

// The ternary super-block tq2_0 is NOW front-door CONSTRUCTED: the front door rewrites the
// abstract tcrv_rvv.dequantize_row (format="tq2_0") into the typed streaming region (decode_model
// "tq2_0", qk=256, stride=66), byte-exact to the retired tq2_0 monolith by construction.
// TERNARY-NOT: tcrv_rvv.dequantize_row {{[^_]}}
// TERNARY: tcrv_rvv.typed_dequantize_row_loop_body %{{.*}}, %{{.*}}, %{{.*}} attributes {decode_model = "tq2_0", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64}
// TERNARY: tcrv_rvv.dequantize_row_decode_core %{{.*}}, %{{.*}}, %{{.*}} {decode_model = "tq2_0", qk = 256 : i64, quant_byte_offset = 0 : i64, scale_byte_offset = 64 : i64, weight_block_stride = 66 : i64}
// TERNARY: tcrv_rvv.typed_dequantize_row_loop_yield
