// RUN: weft-opt %s --weft-rvv-materialize-quantize-row-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: weft-opt %s --weft-rvv-materialize-quantize-row-stream-front-door --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: weft-opt %S/rvv-to-emitc-ggml-dequantize-row-q8-0.mlir --weft-rvv-materialize-quantize-row-stream-front-door | FileCheck %s --check-prefix=DISPATCH

// CERT-FD次族 (quant×3) -- the PRE-EMITC quant-stream FRONT DOOR (the f32->QUANT MIRROR
// of the dequant-stream front door). It runs the same family-local typed formula and
// realization used by the mandatory project-wide pre-emission cut: it rewrites each
// abstract per-format
// weft_rvv.quantize_row_q8_{0,1,K} into the typed weft_rvv.typed_quantize_row_loop_body
// region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }
// and STOPS -- BEFORE --weft-rvv-lower-to-emitc. This exposes the constructed region so
// the certification walker (e5_strong_readout.py stamp-quant-stream) can WALK (and hence
// machine-certify) the realized typed region, instead of it being built-and-erased
// inside a later emission step. The emitter has no abstract-op construction fallback.
// The quant front door is SCOPED to the 3 quantize ops: a
// weft_rvv.dequantize_row is NOT a quantize op, so the pass LEAVES it abstract (DISPATCH
// run below, feeding the dequant fixture). NOTE q8_K here is the ROW quantizer
// (quantize_row_q8_K, the scalar row-quant stream), NOT the mat-quant GEMM path.
// Numerical semantics: zero change (byte-exact by construction).

module {
  weft.exec.kernel @quantize_row_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @quantize_row_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %q = weft_rvv.quantize_row_q8_0 %x, %vy, %n, %vl {kind = "ggml_quantize_row_q8_0", qk = 32 : i64, block_stride = 34 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The front door CONSTRUCTS the typed streaming region and STOPS pre-emitc: the abstract
// weft_rvv.quantize_row_q8_0 is GONE, replaced by the typed body carrying the encode core + yield.
// REALIZE-NOT: weft_rvv.quantize_row_q8_0
// REALIZE-NOT: emitc.
// REALIZE: weft_rvv.typed_quantize_row_loop_body %{{.*}}, %{{.*}}, %{{.*}} attributes {block_stride = 34 : i64, encode_model = "q8_0", kind = "typed_quantize_row_loop_body", qk = 32 : i64, quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>}
// REALIZE: ^bb0(%[[BI:.*]]: index):
// REALIZE: weft_rvv.quantize_row_encode_core %{{.*}}, %{{.*}}, %[[BI]] {block_stride = 34 : i64, encode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>, scale_byte_offset = 0 : i64}
// REALIZE: weft_rvv.typed_quantize_row_loop_yield

// Front door THEN emitc: emission is driven by the typed leaf and consumes the
// formula-produced layout; the provenance token proves the abstract op went through
// the typed region.
// EMIT-NOT: weft_rvv.
// EMIT: emitc.func @weft_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(
// EMIT: route_source_op=weft_rvv.typed_quantize_row_loop_body
// The amax reduction, the f32->i16->i8 RNE narrow, and the int8 qs store (ggml's exact RVV method).
// EMIT: call_opaque "__riscv_vfredmax_vs_f32m8_f32m1"
// EMIT: call_opaque "__riscv_vfncvt_x_f_w_i16m4"
// EMIT: call_opaque "__riscv_vncvt_x_x_w_i8m2"
// EMIT: call_opaque "__riscv_vse8_v_i8m2"

// The quant front door is SCOPED to the 3 quantize ops: a weft_rvv.dequantize_row is NOT a
// quantize op, so the pass LEAVES its abstract op untouched (it lowers via the dequant path)
// and constructs NO quantize region.
// DISPATCH: weft_rvv.dequantize_row %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {format = "q8_0"}
// DISPATCH-NOT: weft_rvv.typed_quantize_row_loop_body
