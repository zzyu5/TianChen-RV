// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq3_xxs` block DECODE (block_iq3_xxs -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the IQ grid-table dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq3_xxs") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq3_xxs", qk=256, stride=98);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string).
//
// PR-31 (the dequant true-vector emitter, first cell): iq3_xxs is the FIRST IQ grid
// format lowered to the OWNED REAL-VECTOR body (emitDequantizeRowIQ3XXSVectorBody) --
// NOT the shared scalar dispatch-wired monolith the other IQ formats still forward to.
// The leaf now emits OWNED __riscv_v intrinsics (the ISSUE-001 reverse: the vector
// content is the emitter's, not host-autovec codegen-lottery): the SAME vluxei16 grid
// gather + sign fold (vmv/vand/vmsne/vneg/vmerge) idiom the iq3_xxs block-dot vec_dot
// body renders, but with the widening-dot tail replaced by an int->float convert
// (vsext_vf4 + vfcvt_f_x_v) + a runtime `db` scale (vfmul_vf) + a unit store (vse32).
// The 8-lane group geometry is the fixed iq3_xxs grid-of-4 x 2 structure (NOT a tunable
// knob). The grid-of-4 (uint32) codebook + the ksigns selector plane + the {1<<j} kmask
// are DERIVED at emit as function-local statics (NO ksigns op-attr; that blocker is
// block-dot-repack-only). Byte-exact-vs-ggml-reference dequantize_row_iq3_xxs by
// construction: the only rounding is db*(float)grid and the sign fold multiplies by an
// EXACT +-1.0f (a float sign flip is bitwise-exact); all grid bytes are < 128 so the
// signed i8 view == ggml's (const uint8_t *) read.

module {
  weft.exec.kernel @dequant_iq3_xxs_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq3_xxs attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq3_xxs, sew = 32 : i64, source_kernel = "dequant_iq3_xxs_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq3_xxs"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq3_xxs_kernel_dequant_iq3_xxs(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The grid-of-4 (uint32) codebook decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq3xxs_grid
// The ksigns selector plane decl, emitted once above the super-block loop.
// CHECK: weft_iq3xxs_ksigns
// The wide sign-fold statics (the {1<<j} pattern replicated across the sub-block's 4
// groups + the per-group broadcast index), each loaded ONCE above the loop at vl=32.
// CHECK: weft_iq3xxs_kmask32
// CHECK: weft_iq3xxs_sigspread
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED real-vector decode (PR-31 + the batch-wide-gather opt): ONE wide vl=8
// vluxei16 grid gather per sub-block + ONE wide vl=32 sign fold (the 4 per-group sign
// bytes broadcast to their 8-lane windows by a vluxei8 spread gather over sigspread,
// then vand+vmsne) + the int->float convert + runtime db scale + full-LMUL store --
// the ISSUE-001 reverse, the fractional-LMUL vl=2 gather storm hoisted.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i32m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_i32m2_i8m2"
// CHECK: call_opaque "__riscv_vluxei8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vv_u8m2"
// CHECK: call_opaque "__riscv_vmerge_vvm_i8m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
