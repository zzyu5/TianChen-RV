// RUN: weft-opt %s --weft-rvv-materialize-dequantize-row-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: weft-opt %s --weft-rvv-materialize-dequantize-row-stream-front-door --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's/"q8_0"/"tq2_0"/g' %s | weft-opt --weft-rvv-materialize-dequantize-row-stream-front-door | FileCheck %s --check-prefix=TERNARY

// CERT-FD首族 (dequant×24) -- the PRE-EMITC dequant-stream FRONT DOOR. It runs ONLY
// the CONSTRUCTION half of constructOrEmitGgmlDequantizeRow (the shared byte-exact
// weft::rvv::constructTypedDequantizeRowLoopBody): it rewrites the abstract
// weft_rvv.dequantize_row into the typed weft_rvv.typed_dequantize_row_loop_body region
//   { dequantize_row_decode_core; typed_dequantize_row_loop_yield }
// and STOPS -- BEFORE --weft-rvv-lower-to-emitc. This exposes the constructed region so
// the certification walker (e5_strong_readout.py stamp-dequant-stream) can WALK (and hence
// machine-certify) the realized typed region, instead of it being built-and-erased atomically
// inside the emitc lowering where no pre-emitc dump can see it.
//
// The construction is the SAME one the in-emitc fallback runs, so the emitted C is byte-exact
// whether the region is built here (pre-emitc, REALIZE) or in emitc: the EMIT run below (front
// door THEN --weft-rvv-lower-to-emitc) is byte-identical to the atomic
// `--weft-rvv-lower-to-emitc`-only path locked by rvv-to-emitc-ggml-dequantize-row-q8-0.mlir
// (the 0-diff is verified format-by-format across all 24 constructed formats out-of-band).
// The ternary super-blocks (tq1_0/tq2_0) and the flat 1-bit binary-sign leaf (q1_0) are NOW
// front-door CONSTRUCTED too: the front door rewrites their abstract op into the SAME typed
// region (TERNARY run below), completing the whole 24-format dequantize_row spectrum -- NO
// format remains dispatch-wired. Numerical semantics: zero change (byte-exact by construction).

module {
  weft.exec.kernel @dequant_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q8_0, sew = 32 : i64, source_kernel = "dequant_q8_0_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q8_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The front door CONSTRUCTS the typed streaming region and STOPS pre-emitc: the abstract
// weft_rvv.dequantize_row is GONE, replaced by the typed body carrying the decode core + yield.
// REALIZE-NOT: weft_rvv.dequantize_row {{[^_]}}
// REALIZE-NOT: emitc.
// REALIZE: weft_rvv.typed_dequantize_row_loop_body %{{.*}}, %{{.*}}, %{{.*}} attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64}
// REALIZE: ^bb0(%[[BI:.*]]: index):
// REALIZE: weft_rvv.dequantize_row_decode_core %{{.*}}, %{{.*}}, %[[BI]] {carrier_kind = "bare_int8", decode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64}
// REALIZE: weft_rvv.typed_dequantize_row_loop_yield

// Front door THEN emitc == the atomic construct+emit path: the emit is DRIVEN by the typed
// region (the provenance token proves the abstract op went THROUGH the typed region), and the
// C is byte-identical to the atomic q8_0 path.
// EMIT-NOT: weft_rvv.
// EMIT: emitc.func @weft_emitc_dequant_q8_0_kernel_dequant_q8_0(
// EMIT: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: !emitc.opaque<"const int8_t">
// The CONSTRUCTED q8_0 path lowers to the OWNED REAL-VECTOR body (PR-31 non-grid cell):
// vle8 + vsext_vf4 + vfcvt_f_x_v + vfmul_vf + vse32, NO gather. Identical to the atomic
// --weft-rvv-lower-to-emitc path locked in rvv-to-emitc-ggml-dequantize-row-q8-0.mlir.
// EMIT: call_opaque "__riscv_vle8_v_i8m2"
// EMIT: call_opaque "__riscv_vsext_vf4_i32m8"
// EMIT: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// EMIT: call_opaque "__riscv_vfmul_vf_f32m8"
// EMIT: call_opaque "__riscv_vse32_v_f32m8"

// The ternary super-block tq2_0 is NOW front-door CONSTRUCTED: the front door rewrites the
// abstract weft_rvv.dequantize_row (format="tq2_0") into the typed streaming region (decode_model
// "tq2_0", qk=256, stride=66), byte-exact to the retired tq2_0 monolith by construction.
// TERNARY-NOT: weft_rvv.dequantize_row {{[^_]}}
// TERNARY: weft_rvv.typed_dequantize_row_loop_body %{{.*}}, %{{.*}}, %{{.*}} attributes {decode_model = "tq2_0", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64}
// TERNARY: weft_rvv.dequantize_row_decode_core %{{.*}}, %{{.*}}, %{{.*}} {decode_model = "tq2_0", qk = 256 : i64, quant_byte_offset = 0 : i64, scale_byte_offset = 64 : i64, weight_block_stride = 66 : i64}
// TERNARY: weft_rvv.typed_dequantize_row_loop_yield
