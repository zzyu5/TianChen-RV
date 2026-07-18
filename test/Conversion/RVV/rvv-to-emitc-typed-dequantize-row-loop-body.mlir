// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_dequantize_row_loop_body"/kind = "plain_dequant_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/"q8_0"/"bf16"/g' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADDECODE

// G3 line-B dequant FRONT DOOR -- the streaming CONSTRUCTED sibling of the flat
// block-dot loop scaffold. weft_rvv.typed_dequantize_row_loop_body carries the ggml
// dequantize_row decode's `nb = k / QK` block loop as ONE region-carrying op whose
// SOLE entry argument is the block_index induction variable (unlike the block-dot
// loop op there is NO loop-carried accumulator: the decode STORES each f32 straight
// through the output pointer, so the region is terminated by the VOID
// weft_rvv.typed_dequantize_row_loop_yield). The per-block decode is the separate
// typed brick weft_rvv.dequantize_row_decode_core (decode_model "q8_0"): read the
// fp16 block scale via the (float)*(const _Float16 *) seam, then decode the 32 signed
// int8 quants as y[j] = qs[j] * d.
//
// This is the FAMILY-HEAD of the 23-format dequantize_row spectrum: the shared
// front-door construction path is proven ONCE here (abstract dequantize_row ->
// typed region -> C); subsequent formats swap only the decode leaf. The CONSTRUCTED
// q8_0 path now lowers to the OWNED REAL-VECTOR body (PR-31, the first NON-GRID cell
// of the dequant true-vector emitter, emitDequantizeRowQ8_0VectorBody): a single
// 32-lane vle8 + vsext_vf4 + vfcvt_f_x_v + vfmul_vf + vse32 pipeline per block, NO
// gather (q8_0 is non-grid) -- NOT the scalar per-element loop the dispatch-wired
// monolith fallback (emitGgmlDequantizeRow bareInt8 branch) still runs. The vector
// content is the EMITTER's OWNED __riscv_v intrinsics (the ISSUE-001 reverse), not
// host-autovec codegen-lottery. Byte-exact-vs-ggml-reference dequantize_row_q8_0 by
// construction: vfmul_vf(qf, d) == the scalar `qs[j]*d` (a single f32 round-to-nearest
// -even multiply; q8_0 has no add/min so no fp-contraction ambiguity). Numerical
// bit-exact-vs-ggml is PROVEN on ssh rvv via the tools/bench/cells/dequantize_row.sh
// harness (ZERO-MODEL byte-exact GREEN, 3-arm anti-hollow); it is not tested here.

module {
  weft.exec.kernel @dequant_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q8_0, sew = 32 : i64, source_kernel = "dequant_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          // The per-block decode leaf (decode_model "q8_0"): the whole per-block
          // scalar decode is emitter-inlined by the brick lowering (the streaming
          // analog of q1_0/nvfp4's flat single-core-brick emit).
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q8_0_kernel_dequant_q8_0(
// The AoS block count nb = k / 32 + the block loop.
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: for %{{.*}} = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// The fp16 block scale seam (fcvt.s.h).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The signed-int8 quant base (the load sign-extends).
// CHECK: !emitc.opaque<"const int8_t">
// The OWNED REAL-VECTOR 32-lane decode pipeline (PR-31 non-grid cell): vle8 (the 32
// int8 quants) + vsext_vf4 (int8->int32) + vfcvt_f_x_v (int32->f32) + vfmul_vf (the
// runtime `d` scale) + vse32 (the contiguous 32-float store). NO gather.
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// CHECK: return

// The bounded surface is fail-closed on the loop kind and the decode_model leaf (I7).
// BADKIND: currently supports only kind "typed_dequantize_row_loop_body"
// BADDECODE: is not a CONSTRUCTED dequantize_row decode
