// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q8_0` block DECODE (block_q8_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the dequant FRONT DOOR (family-head of the
// 23-format spectrum, G3 line-B): the abstract weft_rvv.dequantize_row (format="q8_0")
// is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites it into the
// typed weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the
// typed region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract
// format string).
//
// PR-31 (the dequant true-vector emitter, first NON-GRID cell): the CONSTRUCTED q8_0
// path lowers to the OWNED REAL-VECTOR body (emitDequantizeRowQ8_0VectorBody) -- a
// single 32-lane vle8 (the 32 signed int8 quants) + vsext_vf4 (int8->int32) + vfcvt_f_x_v
// (int32->f32) + vfmul_vf (the runtime `d` scale) + vse32 (contiguous 32-float store)
// pipeline per block, NO gather (q8_0 is non-grid: no codebook, no sign plane, no nibble
// unpack). The leaf emits OWNED __riscv_v intrinsics (the ISSUE-001 reverse: the
// vector content is the emitter's, not host-autovec codegen-lottery), and no parallel
// dispatch-wired monolith remains. The other dequantize_row formats use their typed
// mechanism plans through the same construction cut. The dedicated typed-region -> C
// lowering contract (+ verifier fail-closed) is
// locked in rvv-to-emitc-typed-dequantize-row-loop-body.mlir. Byte-exact-vs-ggml-reference
// dequantize_row_q8_0 by construction: the fp16 d seam is the SAME (float)*(const _Float16 *)
// read, the signed i8 quants sign-extend exactly, and vfmul_vf(qf, d) == the scalar
// `qs[j]*d` (q8_0 has no add/min => no fp-contraction ambiguity).

module {
  weft.exec.kernel @dequant_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q8_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q8_0_kernel_dequant_q8_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body,
// not the dispatch-wired monolith).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS block count + the block loop.
// CHECK: div
// CHECK: for
// The fp16 block scale seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The signed-int8 quant base (the load sign-extends).
// CHECK: !emitc.opaque<"const int8_t">
// The OWNED REAL-VECTOR 32-lane decode pipeline (PR-31 non-grid cell), NO gather.
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
