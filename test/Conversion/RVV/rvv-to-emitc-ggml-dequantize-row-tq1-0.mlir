// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_tq1_0` super-block DECODE (block_tq1_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the ternary super-block dequant leaf (a dequantize_row
// family member, family-head q8_0): the abstract weft_rvv.dequantize_row (format="tq1_0")
// is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core (decode_model
// "tq1_0", qk=256, stride=54); typed_dequantize_row_loop_yield } and lowers it (the emission
// is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8] construction, NOT the
// abstract format string). It lowers via the OWNED REAL-VECTOR body
// emitDequantizeRowTernaryVectorBody (B线批3 ternary de-lottery): byte-exact-vs-ggml-reference
// dequantize_row_tq1_0 by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith keeps the
// scalar emitGgmlDequantizeRowExtended): the fp16 d scale via the (float)*(const _Float16 *)
// seam (@52, block END), then the base-3 unpack -- vle8 the packed qs/qh bytes, vmul_vx by the
// immediate pow3[n] (u8m1, mod-256 truncation == ggml's uint8 `qs*pow3[n]`), vzext_vf2 -> u16m2,
// vmul_vx(3)/vsrl_vx(8) the xi = ((uint16_t)q*3)>>8 extraction, reinterpret u16->i16, vsext_vf2
// -> i32m4, vsub_vx(1), int->float (vfcvt), scaled by d in ONE vfmul (single-mul, no
// fp-contraction ambiguity), stored (vse32) over the qs (160+80) and qh (16) sections. The
// pow3 weights are immediate scalars (NO table). PURE ARITHMETIC decode, NO codebook and NO
// gather. ISSUE-001 reverse; closes the ISSUE-002 codegen-lottery for tq1_0 dequant.

module {
  weft.exec.kernel @dequant_tq1_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_tq1_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_tq1_0, sew = 32 : i64, source_kernel = "dequant_tq1_0_kernel", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "tq1_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_tq1_0_kernel_dequant_tq1_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The block count nb = k / 256 and the block loop.
// CHECK: div
// CHECK: for
// The fp16 d scale seam (@52, block END).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR base-3 ternary decode (B线批3 ternary fan-out): the packed bytes load,
// the vmul_vx by the immediate pow3[n] (mod-256 u8m1), the vzext to u16, the (q*3)>>8 extract
// (vmul_vx / vsrl_vx, u16m2), the reinterpret to signed, the vf2 sign-extend, the subtract-1,
// the int->float convert, the single vfmul by d, and the store. PURE ARITHMETIC -- NO codebook
// table, NO gather. The vector content is the EMITTER's (OWNED __riscv_v), not host-autovec.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vmul_vx_u8m1"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m2"
// CHECK: call_opaque "__riscv_vmul_vx_u16m2"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u16m2_i16m2"
// CHECK: call_opaque "__riscv_vsext_vf2_i32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
