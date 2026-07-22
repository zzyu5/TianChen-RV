// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q4_0` block DECODE (block_q4_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the flat nibble dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q4_0") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q4_0"); typed_dequantize_row_loop_yield }
// and lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q4_0 monolith modulo ONLY the source-op
// provenance token (the shared body emitter emitDequantizeRowNibbleBodyShared, reached
// via emitDequantizeRowQ4_0BodyShared, is common to both) -- the fp16 block scale via
// the (float)*(const _Float16 *) seam, then the nibble unpack (qs[j]&0x0F)-8 -> y[j]
// and (qs[j]>>4)-8 -> y[j+16]. The q4_0 nibble decode REUSES the same block-decode
// already built for the q4_0 block-dot vec_dot. Byte-exact-vs-ggml-reference
// dequantize_row_q4_0 (a scalar AoS block loop; no reduction).

module {
  weft.exec.kernel @dequant_q4_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q4_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q4_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q4_0_kernel_dequant_q4_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body,
// not the dispatch-wired monolith).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS block count nb = k / 32 and the block loop.
// CHECK: div
// CHECK: for
// The fp16 block scale seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR nibble decode (R线 §四.1 SAFE set, PR-31 fan-out): the 16 packed
// bytes load, the low/high nibble split (vand_vx / vsrl_vx), the vf4 widen, the -8 bias,
// the int->float convert, and the SINGLE-mul `d` scale (vfmul_vf, NO fp-contraction). NO
// gather. The vector content is the EMITTER's (OWNED __riscv_v), not host-autovec lottery.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
