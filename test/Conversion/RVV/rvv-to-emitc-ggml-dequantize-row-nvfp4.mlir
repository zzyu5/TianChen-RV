// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_nvfp4` block DECODE (block_nvfp4 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the FP4 codebook dequant leaf (G3 line-B negative
// control, a dequantize_row family member, family-head q8_0): the abstract
// weft_rvv.dequantize_row (format="nvfp4") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "nvfp4", qk=64, stride=36); typed_dequantize_row_loop_yield } and lowers
// it. Source format/decode_model carry construction identity, coherence, and provenance
// only; post-construction emission is driven by typed mechanism/scale g plus the complete
// selected codebook-gather stamp, NOT the abstract format string. It lowers via the OWNED REAL-VECTOR body
// emitDequantizeRowCodebookVectorBody (B线批2 tiny-codebook fan-out): byte-exact-vs-ggml-
// reference dequantize_row_nvfp4 by CONSTRUCTION (NOT byte-identical to the scalar monolith
// -- the CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith
// keeps the scalar emitGgmlDequantizeRowExtended): the four per-16-element UE4M3 sub-block
// scales (ggml_ue4m3_to_fp32 via ldexpf, HALF form) in SCALAR C, then per sub the FP4
// codebook (shared with mxfp4, broadcast into the i8m1 anchor selected for this fixture's
// VLEN128 c) GATHERED by the two nibble
// index lanes (vrgather_vv_i8m1 -- a REGISTER-RESIDENT codebook gather, NOT a vluxei memory
// gather, so NO HW-gather wall), sign-extended (vsext_vf4), int->float (vfcvt), scaled by
// the sub scale in ONE vfmul (single-mul, no fp-contraction ambiguity), stored (vse32). The
// codebook + UE4M3 structure is selected/stamped before emission; the emitter
// mechanically materializes it. ISSUE-001 reverse; closes the ISSUE-002
// codegen-lottery for nvfp4 dequant.

module {
  weft.exec.kernel @dequant_nvfp4_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8", supported_sew = "8,16,32,64"}
    weft.exec.variant @dequant_nvfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_nvfp4, sew = 32 : i64, source_kernel = "dequant_nvfp4_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "nvfp4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_nvfp4_kernel_dequant_nvfp4(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The FP4 codebook table decl (kvalues_mxfp4, reused by nvfp4), above the loop, then
// broadcast into the selected i8m1 vreg for this VLEN128 fixture (register-resident,
// reused by every vrgather).
// CHECK: static const int8_t weft_dequant_mxfp4_kvalues
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The block count nb = k / 64 and the block loop.
// CHECK: div
// CHECK: for
// The UE4M3 -> fp32 HALF scale via ldexpf (NO fp16 seam), in SCALAR C.
// CHECK: call_opaque "ldexpf"
// The OWNED REAL-VECTOR codebook nibble decode (B线批2 tiny-codebook fan-out): the packed
// nibble bytes load, the low/high nibble split (vand_vx / vsrl_vx), the REGISTER codebook
// gather (vrgather_vv_i8m1, NOT vluxei memory gather), the vf4 sign-extend, the int->float
// convert, the single vfmul by the sub scale, and the store. OWNED __riscv_v, not lottery.
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
