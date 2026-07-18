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
// PR-31 / candidate ② (裁决3 / ISSUE-107): iq3_xxs is the FIRST IQ grid format lowered
// to an OWNED body (emitDequantizeRowIQ3XXSVectorBody) -- NOT the shared scalar dispatch-
// wired monolith the other IQ formats still forward to. Candidate ② emits the OPPONENT
// SHAPE: the deployed dequantize_row_iq3_xxs has HW_GATHER=0 (objdump: scalar grid-index
// loads + unit-stride vector arithmetic, NO vluxei/vlox indexed gather). This leaf reads
// the 8 grid u32 entries by SCALAR indexed loads grid32[qg[s]] into a stack gstage[8] +
// ONE unit-stride vle32, and SCALAR-spreads the 4 per-group sign bytes into a stack
// sigstage[32] + ONE unit-stride vle8 -- then the SAME full-LMUL sign fold (vand/vmsne/
// vneg/vmerge) + int->float convert (vsext_vf4 + vfcvt_f_x_v) + runtime `db` scale
// (vfmul_vf) + unit store (vse32). It emits OWNED __riscv_v intrinsics (the ISSUE-001
// reverse: the vector content is the emitter's, not host-autovec codegen-lottery; the
// emitter DETERMINES the scalar-load structure, de-lottery [L-8]) -- but with NO HW
// indexed gather (the grid HW-gather variant hit the 0.36 vluxei ceiling, ISSUE-107).
// The 8-lane group geometry is the fixed iq3_xxs grid-of-4 x 2 structure (NOT a tunable
// knob). The grid-of-4 (uint32) codebook + the ksigns selector plane + the {1<<j} kmask
// are DERIVED at emit as function-local statics (NO ksigns op-attr; that blocker is
// block-dot-repack-only). Byte-exact-vs-ggml-reference dequantize_row_iq3_xxs by
// construction: gstage[s] == grid32[qg[s]] and sigstage[l*8+j] == the l-th sign byte
// (only the transport into the vector registers differs from the HW-gather variant); the
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
// The grid-of-4 (uint32) codebook decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq3xxs_grid
// The ksigns selector plane decl, emitted once above the super-block loop.
// CHECK: weft_iq3xxs_ksigns
// The {1<<j} sign-bit selector static, replicated across the sub-block's 4 groups and
// loaded ONCE above the loop at vl=32 (candidate ② has NO sigspread broadcast-index
// table -- the sign bytes are SCALAR-spread into a stack sigstage[32]).
// CHECK: weft_iq3xxs_kmask32
// CHECK-NOT: weft_iq3xxs_sigspread
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED candidate ② decode (opponent shape, HW_GATHER=0): the 8 grid u32 entries
// read by SCALAR indexed loads grid32[qg[s]] into gstage[8] + ONE unit-stride vle32, the
// 4 per-group sign bytes SCALAR-spread into sigstage[32] + ONE unit-stride vle8, then the
// SAME vl=32 sign fold (vand+vmsne+vneg+vmerge) + int->float convert + runtime db scale +
// full-LMUL store -- the ISSUE-001 reverse WITHOUT any __riscv_vluxei/vlox indexed gather.
// CHECK: call_opaque "__riscv_vle32_v_i32m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_i32m2_i8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vv_u8m2"
// CHECK: call_opaque "__riscv_vmerge_vvm_i8m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m8"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// candidate ② is byte-exact to the grid HW-gather variant but transports the grid/sign
// bytes via scalar loads + unit-stride vle -- NO HW indexed gather anywhere in the leaf.
// CHECK-NOT: vluxei
// CHECK-NOT: vloxei
// CHECK-NOT: vrgather
