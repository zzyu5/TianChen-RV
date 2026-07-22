// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq2_s` block DECODE (block_iq2_s -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the IQ grid-table dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq2_s") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq2_s", qk=256, stride=82);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). iq2_s dequant now FANS OUT to its OWNED narrow-per-entry REAL-VECTOR body
// (emitDequantizeRowIQ2SVectorBody, W5 grid family de-lottery -- the iq2_xs grid-of-8
// sibling with EXPLICIT sign bytes): each 8-value grid entry (grid-of-8 int64, 1024-entry)
// decoded by ONE narrow OWNED pipeline over its 8 CONTIGUOUS grid bytes -- a SCALAR-computed
// grid pointer (gridb + idx*8 where idx = qs[l]|((qh[ib32]<<(8-2l))&0x300), NO indexed
// gather) + a unit-stride vle8 (vl=8) + the per-lane +-1 sign applied by an INTEGER vmul_vv
// against the UNIVERSAL signs256 +-1 plane indexed by the EXPLICIT 8-bit sign byte (8
// CONTIGUOUS +-1 bytes, also a unit-stride vle8) + vsext_vf4->i32m2 + vfcvt + vfmul(db) +
// vse32, db=d*(0.5+scale_nibble)*0.25. The 1024-entry (int64) grid + the signs256 plane are
// DERIVED at emit as function-local statics (NO signs op-attr; that blocker is
// block-dot-repack-only, not this streaming dequant path; REUSES the iq2_s block-dot vec_dot
// grid + signs256 decls) and the fp16 d arrives via the (float)*(const _Float16 *) seam.
// This REPLACES the prior scalar-AoS forwarder that clang's -O3 autovec blows into a
// HW-gather codegen-lottery leaf (ISSUE-001/002); gather-free. Byte-exact-vs-ggml-reference
// by construction (grid[j]*sign EXACT integer, grid bytes < 128 so signed i8 view == ggml's
// (const uint8_t *) read; float sign-flip and mul-by-+-1 are bitwise-exact; no reduction).

module {
  weft.exec.kernel @dequant_iq2_s_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq2_s attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq2_s"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq2_s_kernel_dequant_iq2_s(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The 1024-entry (int64) grid decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq2s_grid
// The universal signs256 sign plane decl, emitted once above the super-block loop.
// CHECK: weft_iq2s_signs256
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED narrow-per-entry real-vector pipeline (gather-free): unit-stride vle8 of the 8
// CONTIGUOUS grid bytes + the signs256 +-1 bytes, the INTEGER sign-fold vmul, the m2 widen /
// convert / db-scale, and the unit store. NO __riscv_vluxei / vslidedown (the flip).
// CHECK: __riscv_vle8_v_i8mf2
// CHECK: __riscv_vmul_vv_i8mf2
// CHECK: __riscv_vsext_vf4_i32m2
// CHECK: __riscv_vfcvt_f_x_v_f32m2
// CHECK: __riscv_vfmul_vf_f32m2
// CHECK: __riscv_vse32_v_f32m2
// CHECK-NOT: __riscv_vluxei
// CHECK-NOT: __riscv_vloxei
