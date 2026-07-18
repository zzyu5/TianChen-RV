// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q5_0` block DECODE (block_q5_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the flat nibble+5th-bit dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="q5_0") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "q5_0"); typed_dequantize_row_loop_yield }
// and lowers it (the emission is DRIVEN by the typed region op-identity + decode_model,
// [L-6]/[L-8] construction, NOT the abstract format string). The emitted C is
// BYTE-IDENTICAL to the dispatch-wired q5_0 monolith modulo ONLY the source-op
// provenance token (the shared body emitter emitDequantizeRowNibbleBodyShared, reached
// via emitDequantizeRowQ5_0BodyShared, is common to both) -- the fp16 scale via the
// (float)*(const _Float16 *) seam, the byte-assembled little-endian uint32 qh 5th-bit
// plane (qh[0] | qh[1]<<8 | qh[2]<<16 | qh[3]<<24, matching ggml's memcpy(&qh)), then
// ((qs[j]&0x0F)|xh0)-16 -> y[j] and ((qs[j]>>4)|xh1)-16 -> y[j+16]. The q5_0 qh merge
// REUSES the block-decode already built for the q5_0 block-dot vec_dot.
// Byte-exact-vs-ggml-reference dequantize_row_q5_0 (a scalar AoS block loop).

module {
  weft.exec.kernel @dequant_q5_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_q5_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q5_0, sew = 32 : i64, source_kernel = "dequant_q5_0_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "q5_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_q5_0_kernel_dequant_q5_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The AoS block count + the block loop.
// CHECK: div
// CHECK: for
// The fp16 block scale seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The byte-assembled uint32 qh 5th-bit plane (shift the qh bytes into place + OR).
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// The OWNED REAL-VECTOR nibble decode (R线 §四.1 SAFE set, PR-31 fan-out): the 16 packed
// bytes load, the low/high nibble split, the vf4 widen, then the 5th-bit spread
// (vid / vmv broadcast qh / vsrl_vv per-lane / vand / vsll / vor -> {0,16}), the -16 bias,
// the int->float convert, and the SINGLE-mul `d` scale (vfmul_vf, NO fp-contraction). NO
// gather. The vector content is the EMITTER's (OWNED __riscv_v), not host-autovec lottery.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf4_u32m4"
// CHECK: call_opaque "__riscv_vid_v_u32m4"
// CHECK: call_opaque "__riscv_vmv_v_x_u32m4"
// CHECK: call_opaque "__riscv_vsrl_vv_u32m4"
// CHECK: call_opaque "__riscv_vsll_vx_u32m4"
// CHECK: call_opaque "__riscv_vor_vv_u32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
