// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq3_s` block DECODE (block_iq3_s -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the IQ grid-table dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq3_s") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq3_s", qk=256, stride=110);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). iq3_s dequant now FANS OUT to its OWNED narrow-per-entry REAL-VECTOR body
// (emitDequantizeRowIQ3SVectorBody, R5.1-C grid family de-lottery -- the iq3_xxs sibling):
// each 4-byte grid entry decoded by ONE narrow OWNED pipeline over its 4 CONTIGUOUS grid
// bytes -- a SCALAR-computed grid pointer (gridb + idx*4, idx = q | ((qh<<k)&256) with the
// 9th index bit merged from qh) + a unit-stride vle8 (vl=4, NO indexed gather) + an i8 sign
// fold from the EXPLICIT sign byte (entry1 = kmask_lo{1,2,4,8}, entry2 = kmask_hi{16,32,64,
// 128}) + vsext_vf4->i32m1 + vfcvt + vfmul(db) + vse32, db=d*(1+2*scale). The 512-entry
// (uint32) grid-of-4 is DERIVED at emit as a function-local static (NO grid op-attr; REUSES
// the iq3_s block-dot vec_dot grid decl) and the fp16 d arrives via the (float)*(const
// _Float16 *) seam. This REPLACES the prior scalar-AoS forwarder that clang's -O3 autovec
// blew up into a vluxei16 HW-gather / vslidedown codegen-lottery leaf (ISSUE-001/002);
// board-proven gather-free ~2x that scalar leaf and beats ggml dequantize_row_iq3_s.
// Byte-exact-vs-ggml-reference by construction (only rounding db*(float)grid; sign fold is
// EXACT +-1.0f; all grid bytes < 16 so the signed i8 view == ggml's (const uint8_t *) read;
// no reduction).

module {
  weft.exec.kernel @dequant_iq3_s_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq3_s attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq3_s"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The 512-entry (uint32) grid-of-4 decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq3s_grid
// The two explicit-sign selectors ({1,2,4,8} / {16,32,64,128}), emitted once above the loop.
// CHECK: weft_iq3s_kmask_lo
// CHECK: weft_iq3s_kmask_hi
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED narrow-per-entry decode (de-lottery guard: a scalar forwarder has NONE of these
// -- reverting to it would fail here): the unit-stride grid entry load (NOT an indexed
// gather), the i8 sign fold, and the narrow int->float widening + db scale + store.
// CHECK: __riscv_vle8_v_i8mf4
// CHECK: __riscv_vand_vx_u8mf4
// CHECK: __riscv_vmerge_vvm_i8mf4
// CHECK: __riscv_vsext_vf4_i32m1
// CHECK: __riscv_vfcvt_f_x_v_f32m1
// CHECK: __riscv_vfmul_vf_f32m1
// CHECK: __riscv_vse32_v_f32m1
