// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_tq2_0` super-block DECODE (block_tq2_0 -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the ternary super-block dequant leaf (a dequantize_row
// family member, family-head q8_0): the abstract weft_rvv.dequantize_row (format="tq2_0")
// is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites it into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core (decode_model
// "tq2_0", qk=256, stride=66); typed_dequantize_row_loop_yield } and lowers it (the emission
// is DRIVEN by the typed region op-identity + decode_model, [L-6]/[L-8] construction, NOT the
// abstract format string). It lowers via the OWNED REAL-VECTOR body
// emitDequantizeRowTernaryVectorBody (B线批3 ternary de-lottery): byte-exact-vs-ggml-reference
// dequantize_row_tq2_0 by CONSTRUCTION (NOT byte-identical to the scalar monolith -- the
// CONSTRUCTED path now emits OWNED __riscv_v intrinsics, the dispatch-wired monolith keeps the
// scalar emitGgmlDequantizeRowExtended): the fp16 d scale via the (float)*(const _Float16 *)
// seam (@64, block END), then the 2-bit ternary unpack -- vle8 the packed qs bytes, vsrl_vx/
// vand_vx the 2-bit field q = (qs>>(2l))&3 (u8m1), reinterpret u8->i8, vsext_vf4 -> i32m4,
// vsub_vx(1), int->float (vfcvt), scaled by d in ONE vfmul (single-mul, no fp-contraction
// ambiguity), stored (vse32) over the eight (j-block,l) groups (16-lane halves). PURE
// ARITHMETIC decode, NO codebook and NO gather. ISSUE-001 reverse; closes the ISSUE-002
// codegen-lottery for tq2_0 dequant.

module {
  weft.exec.kernel @dequant_tq2_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_tq2_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "tq2_0"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_tq2_0_kernel_dequant_tq2_0(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The block count nb = k / 256 and the block loop.
// CHECK: div
// CHECK: for
// The fp16 d scale seam (@64, block END).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED REAL-VECTOR 2-bit ternary decode (B线批3 ternary fan-out): the packed qs bytes
// load, the 2-bit field split (vsrl_vx / vand_vx, u8m1), the reinterpret to signed, the vf4
// sign-extend, the subtract-1, the int->float convert, the single vfmul by d, and the store.
// PURE ARITHMETIC -- NO codebook, NO gather. The vector content is the EMITTER's (OWNED
// __riscv_v), not host-autovec lottery.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m4"
// CHECK: call_opaque "__riscv_vsub_vx_i32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// The 2-bit shift for the l>0 groups.
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
