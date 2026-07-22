// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq1_m` block DECODE (block_iq1_m -> f32 row) as the
// CONSTRUCT-FROM-ABSTRACT proof of the ternary-grid dequant leaf (G3 line-B, a
// dequantize_row family member, family-head q8_0): the abstract weft_rvv.dequantize_row
// (format="iq1_m") is FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites
// it into the typed weft_rvv.typed_dequantize_row_loop_body region {
// dequantize_row_decode_core (decode_model "iq1_m", qk=256, stride=56);
// typed_dequantize_row_loop_yield } and lowers it (the emission is DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, NOT the abstract format
// string). iq1_m dequant now FANS OUT to its OWNED narrow-per-entry REAL-VECTOR body
// (emitDequantizeRowIQ1MVectorBody, R5.1-D grid family de-lottery [L-8]): each 8-value
// ternary iq1s_grid entry decoded by ONE narrow OWNED pipeline over its 8 CONTIGUOUS grid
// bytes -- a SCALAR grid pointer (gridb + idx*8, NO indexed gather) + a unit-stride vle8
// (vl=8, SIGNED read == ggml's (const int8_t *) read) + vsext_vf4->i32m2 + vfcvt +
// vfadd(delta) + vfmul(dl) + vse32; idx = qs|((qh<<sh)&0x700), dl = d*(2*((sc>>sh)&7)+1),
// delta = (qh & bit)?-0.125:0.125. NOTE (honest-null · lever-N/A): UNLIKE the iq2_xs/iq3_s
// flip, the prior iq1_m scalar forwarder was ALREADY gather-free after clang -O3 autovec
// (board objdump: vlux=0, vslidedown=0 -- grid[j] j=0..7 is a CONTIGUOUS int8 read, the
// narrow-per-entry shape clang lands well), so this OWNED body is board-PARITY (de-lottery
// robustness: own the codegen instead of riding clang's lottery), NOT a codegen-STRUCTURE
// speedup. The 2048-entry ternary iq1m_grid is DERIVED at emit as a function-local static
// (NO grid op-attr; REUSES the iq1_m block-dot vec_dot grid decl); NO fp16 d field (the
// super-block scale is the packed iq1m_scale fp16 reconstructed from the 4 scale words then
// read AS _Float16 via the (float)*(const _Float16 *) seam). Byte-exact-vs-ggml-reference
// dequantize_row_iq1_m by construction (dl*(float(grid)+delta) = one add + one mul, same
// rounding order; no reduction).

module {
  weft.exec.kernel @dequant_iq1_m_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq1_m attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "iq1_m"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_dequant_iq1_m_kernel_dequant_iq1_m(
// The construction is real: the emit is DRIVEN by the typed region (the provenance
// token proves the abstract op went THROUGH weft_rvv.typed_dequantize_row_loop_body).
// CHECK: route_source_op=weft_rvv.typed_dequantize_row_loop_body
// The ternary grid table decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: weft_iq1m_grid
// The super-block loop, then the reconstructed packed-scale fp16 seam inside it.
// CHECK: for %
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The OWNED narrow-per-entry real-vector pipeline (gather-free): unit-stride vle8 of the 8
// CONTIGUOUS ternary grid bytes, the m2 widen / convert, the +-0.125 delta add, the dl
// scale, and the unit store. NO __riscv_vluxei / vslidedown.
// CHECK: __riscv_vle8_v_i8mf2
// CHECK: __riscv_vsext_vf4_i32m2
// CHECK: __riscv_vfcvt_f_x_v_f32m2
// CHECK: __riscv_vfadd_vf_f32m2
// CHECK: __riscv_vfmul_vf_f32m2
// CHECK: __riscv_vse32_v_f32m2
// CHECK-NOT: __riscv_vluxei
// CHECK-NOT: __riscv_vloxei
