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
// W4 schedule-lock rewrite (r5.1-W4, ISSUE-107 wall-type correction): iq3_xxs is the
// FIRST IQ grid format lowered to an OWNED body (emitDequantizeRowIQ3XXSVectorBody) --
// NOT the shared scalar dispatch-wired monolith the other IQ formats still forward to.
// The prior candidate ② staged the 8 grid u32 entries by SCALAR array-indexed loads
// grid32[qg[s]] into a stack gstage[8] + ONE wide vle32, then a full-LMUL vl=32 fold; clang
// -O3 SLP-recognised that staging as a vluxei16 gather idiom and re-vectorised it back into
// 8 HW gathers, capping the cell at 0.36 (mis-recorded as a HW-gather throughput ceiling,
// ISSUE-107). W4 emits the deployed opponent's NARROW shape with OWNED intrinsics: each of
// the sub-block's 8 grid entries is decoded by ONE 4-lane pipeline over its 4 contiguous
// grid bytes -- a SCALAR-computed grid pointer (gridb + idx*4, a sh2add) + a contiguous
// vle8 (vl=4, NO indexed gather) + i8 sign fold (vand_vx/vmsne/vneg/vmerge) + vsext_vf4 ->
// i32m1 + vfcvt_f_x_v + runtime `db` scale (vfmul_vf) + a 4-float vse32. It emits OWNED
// __riscv_v intrinsics (the ISSUE-001 reverse: the vector content is the emitter's, not
// host-autovec codegen-lottery; the emitter DETERMINES the scalar-load structure, de-
// lottery [L-8]). The lever is BOTH the gather-free assembly AND the cheap narrow m1
// widening (the wide m8 vsext/vfcvt was the real cost) -- board-proven 0.36 -> ~1.4 @rvv
// vs the deployed autovec dequantize_row_iq3_xxs (scalar-class tier), byte-exact.
// The 4-lane grid-entry geometry is the fixed iq3_xxs grid-of-4 structure (NOT a tunable
// knob). The grid-of-4 (uint32) codebook + the ksigns selector plane + the two 4-lane
// {1<<j} kmask selectors are DERIVED at emit as function-local statics (NO ksigns op-attr;
// that blocker is block-dot-repack-only). Byte-exact-vs-ggml-reference dequantize_row_iq3_
// xxs by construction: gv == grid[qg[2l+h]] and the sign fold applies the exact +-1.0f from
// the l-th ksigns byte (only the transport into the vector registers differs); the
// only rounding is db*(float)grid and the sign fold multiplies by an EXACT +-1.0f (a
// float sign flip is bitwise-exact); all grid bytes are < 128 so the signed i8 view ==
// ggml's (const uint8_t *) read.

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
// The grid-of-4 (uint32) codebook + the ksigns selector plane, emitted once above the loop.
// CHECK: weft_iq3xxs_grid
// CHECK: weft_iq3xxs_ksigns
// The two 4-lane {1<<j} sign selectors (even grid entry {1,2,4,8}, odd {16,32,64,128}),
// loaded ONCE at vl=4 -- the W4 narrow per-entry decode has NO wide kmask32 and NO
// sigspread broadcast-index table.
// CHECK: weft_iq3xxs_kmask_lo
// CHECK: weft_iq3xxs_kmask_hi
// CHECK-NOT: weft_iq3xxs_kmask32
// CHECK-NOT: weft_iq3xxs_sigspread
// CHECK: call_opaque "__riscv_vle8_v_u8mf4"
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The W4 narrow per-entry OWNED decode (gather-free, board-proven 0.36 -> ~1.4 @rvv): each
// grid entry's 4 bytes are read by a SCALAR-computed pointer (gridb + idx*4, a sh2add) + a
// contiguous vle8 (vl=4, NO indexed gather); the i8 sign fold (vand_vx/vmsne/vneg/vmerge) +
// vsext_vf4 -> i32m1 + vfcvt + runtime db scale + 4-float store is byte-identical to the
// old wide vl=32 fold, but never materialises the wide grid vector clang re-gathers.
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: call_opaque "__riscv_vand_vx_u8mf4"
// CHECK: call_opaque "__riscv_vmsne_vx_u8mf4_b32"
// CHECK: call_opaque "__riscv_vneg_v_i8mf4"
// CHECK: call_opaque "__riscv_vmerge_vvm_i8mf4"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m1"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m1"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m1"
// The emitted leaf transports the grid/sign bytes via scalar-computed pointers + unit-
// stride vle -- NO HW indexed gather anywhere (candidate 2's wide gstage[8]+vle32 staging,
// which clang -O3 re-vectorised into vluxei16, is gone).
// CHECK-NOT: __riscv_vle32_v_i32m2
// CHECK-NOT: vluxei
// CHECK-NOT: vloxei
// CHECK-NOT: vrgather
