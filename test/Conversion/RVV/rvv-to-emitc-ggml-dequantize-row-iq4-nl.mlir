// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq4_nl` block DECODE (block_iq4_nl -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the non-linear codebook dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq4_nl") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq4_nl", qk=32, stride=18);
// typed_dequantize_row_loop_yield } and lowers it. Source format/decode_model carry
// construction identity, coherence, and provenance only; post-construction emission is
// driven by typed mechanism/scale g plus the complete selected codebook-gather stamp, NOT
// the abstract format string. It lowers via the OWNED REAL-VECTOR body emitDequantizeRowCodebookVectorBody
// (B线批2 tiny-codebook fan-out): byte-exact-vs-ggml-reference dequantize_row_iq4_nl by
// CONSTRUCTION (NOT byte-identical to the scalar monolith -- the CONSTRUCTED path now emits
// OWNED __riscv_v intrinsics, the dispatch-wired monolith keeps the scalar
// emitGgmlDequantizeRowExtended): the fp16 d via the (float)*(const _Float16 *) seam, then
// for this fixture's VLEN128 c, pre-emission selection chooses i8m1 and the emitter
// broadcasts the 16-entry non-linear codebook there (vle8_v_i8m1, 16); the
// two nibble index lanes (vand 0x0F / vsrl 0x04) GATHERED through it (vrgather_vv_i8m1 -- a
// REGISTER-RESIDENT codebook gather, NOT a vluxei memory gather, so NO HW-gather wall),
// sign-extended (vsext_vf4), int->float (vfcvt), scaled by d in ONE vfmul (single-mul, no
// fp-contraction ambiguity), stored (vse32). Table/scale structure is selected and
// stamped before emission; the emitter mechanically materializes its function-local
// static. ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for
// iq4_nl dequant.

module {
  weft.exec.kernel @dequant_iq4_nl_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8", supported_sew = "8,16,32,64"}
    weft.exec.variant @dequant_iq4_nl attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq4_nl, sew = 32 : i64, source_kernel = "dequant_iq4_nl_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq4_nl"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq4_nl_kernel_dequant_iq4_nl(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The 16-entry non-linear codebook table decl, emitted once above the loop, then
// broadcast into the selected i8m1 vreg for this VLEN128 fixture (register-resident,
// reused by every vrgather).
// CHECK: static const int8_t weft_dequant_iq4nl_kvalues
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: div
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR codebook nibble decode (B线批2 tiny-codebook fan-out): the 16
// packed nibble bytes load, the low/high nibble split (vand_vx / vsrl_vx), the REGISTER
// codebook gather (vrgather_vv_i8m1, NOT vluxei memory gather), the vf4 sign-extend, the
// int->float convert, the single vfmul by d, and the store. OWNED __riscv_v, not lottery.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
