// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_mxfp4` block DECODE (block_mxfp4 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the FP4 codebook dequant leaf (G3 line-B negative
// control, a dequantize_row family member, family-head q8_0): the abstract
// weft_rvv.dequantize_row (format="mxfp4") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "mxfp4", qk=32, stride=17); typed_dequantize_row_loop_yield } and lowers it
// (the emission is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8]
// construction, NOT the abstract format string). It lowers via the OWNED REAL-VECTOR body
// emitDequantizeRowCodebookVectorBody (B线批2 tiny-codebook fan-out): byte-exact-vs-ggml-
// reference dequantize_row_mxfp4 by CONSTRUCTION (NOT byte-identical to the scalar monolith
// -- the CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith
// keeps the scalar emitGgmlDequantizeRowExtended): the E8M0 block scale reconstructed by
// ggml's exact bit construction (0x00200000u denormal / (e-1)<<23 normal) in SCALAR C, then
// the 16-entry FP4 (e2m1) codebook broadcast into ONE i8m1 vreg (vle8_v_i8m1, 16) and the
// two nibble index lanes (vand 0x0F / vsrl 0x04) GATHERED through it (vrgather_vv_i8m1 -- a
// REGISTER-RESIDENT codebook gather, NOT a vluxei memory gather, so NO HW-gather wall),
// sign-extended (vsext_vf4), int->float (vfcvt), scaled by d in ONE vfmul (single-mul, no
// fp-contraction ambiguity), stored (vse32). The codebook + E8M0 scale are DERIVED at emit.
// ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for mxfp4 dequant.

module {
  weft.exec.kernel @dequant_mxfp4_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_mxfp4, sew = 32 : i64, source_kernel = "dequant_mxfp4_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "mxfp4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The FP4 codebook table decl (kvalues_mxfp4), emitted once above the loop, then
// broadcast into ONE i8m1 vreg (register-resident, reused by every vrgather).
// CHECK: static const int8_t weft_dequant_mxfp4_kvalues
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The block count nb = k / 32 and the block loop.
// CHECK: div
// CHECK: for
// The E8M0 -> fp32 HALF scale bit construction (NO fp16 seam), in SCALAR C.
// CHECK: 0x00200000u
// The OWNED REAL-VECTOR codebook nibble decode (B线批2 tiny-codebook fan-out): the 16
// packed nibble bytes load, the low/high nibble split (vand_vx / vsrl_vx), the
// REGISTER codebook gather (vrgather_vv_i8m1, NOT vluxei memory gather), the vf4
// sign-extend, the int->float convert, the single vfmul by d, and the store. The
// vector content is the EMITTER's (OWNED __riscv_v), not host-autovec lottery.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
