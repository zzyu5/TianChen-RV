// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq4_xs` block DECODE (block_iq4_xs -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the super-block non-linear codebook dequant leaf (G3
// line-B, a dequantize_row family member, family-head q8_0): the abstract
// weft_rvv.dequantize_row (format="iq4_xs") is FRONT-DOOR CONSTRUCTED --
// constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core
// (decode_model "iq4_xs", qk=256, stride=136); typed_dequantize_row_loop_yield } and lowers
// it (the emission is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8]
// construction, NOT the abstract format string). It lowers via the OWNED REAL-VECTOR body
// emitDequantizeRowCodebookVectorBody (B线批2 tiny-codebook fan-out): byte-exact-vs-ggml-
// reference dequantize_row_iq4_xs by CONSTRUCTION (NOT byte-identical to the scalar monolith
// -- the CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith
// keeps the scalar emitGgmlDequantizeRowExtended): the fp16 d via the (float)*(const
// _Float16 *) seam, the 6-bit ls from a scales_l nibble + a scales_h 2-bit (dl=d*(ls-32)) in
// SCALAR C, then per sub-block the 16-entry iq4_nl codebook (broadcast into ONE i8m1 vreg)
// GATHERED by the two nibble index lanes (vrgather_vv_i8m1 -- a REGISTER-RESIDENT codebook
// gather, NOT a vluxei memory gather, so NO HW-gather wall), sign-extended (vsext_vf4),
// int->float (vfcvt), scaled by dl in ONE vfmul (single-mul, no fp-contraction ambiguity),
// stored (vse32). The codebook is DERIVED at emit as a function-local static. ISSUE-001
// reverse; closes the ISSUE-002 codegen-lottery for iq4_xs dequant.

module {
  weft.exec.kernel @dequant_iq4_xs_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq4_xs attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq4_xs, sew = 32 : i64, source_kernel = "dequant_iq4_xs_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq4_xs"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq4_xs_kernel_dequant_iq4_xs(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The codebook table decl, emitted once above the super-block loop (reused from the vec_dot
// codebook), then broadcast into ONE i8m1 vreg (register-resident, reused by every vrgather).
// CHECK: weft_dequant_iq4nl_kvalues
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for %
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR codebook nibble decode (B线批2 tiny-codebook fan-out): the packed
// nibble bytes load, the low/high nibble split (vand_vx / vsrl_vx), the REGISTER codebook
// gather (vrgather_vv_i8m1, NOT vluxei memory gather), the vf4 sign-extend, the int->float
// convert, the single vfmul by dl, and the store. OWNED __riscv_v, not host-autovec lottery.
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
