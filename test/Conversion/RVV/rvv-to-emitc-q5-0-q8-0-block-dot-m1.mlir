// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// wa2 — the COMPLETE ggml ggml_vec_dot_q5_0_q8_0 block kernel lowered at the
// WIDE-LMUL m1 DEFAULT. This input is ATTR-LESS (no integer_core_lmul), so it
// exercises the NEW default the wa2 brick flipped to: the 5-bit nibble+qh
// block-dot default went from the legacy mf4 narrow strip to the m1 whole-half-
// block floor that matches ggml's one-vwredsum reduction (16-lane half-block in
// ONE strip at VLEN>=128, the VLEN-universal robust floor). The hand-authored
// rvv-to-emitc-q5-0-q8-0-block-dot.mlir now pins the explicit mf4 narrow anchor
// (the now-non-default form), so the two diverge by exactly the wa2 LMUL flip.
// The 5th-bit qh injection (emitFiveBitOffsetBinaryDecodeProductValue) is
// width-parametrized on coreLmul/wideLmul, so the wider m1 decode is byte-
// identical to the mf4 narrow one -- the vid+c shift vector indexes the SAME
// per-element qh bit at any VLMAX, the offset-binary `-16` bias is unchanged.
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer AoS block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-block address arithmetic (vx + ib*22, vy + ib*34) -- structured mul/add.
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg2, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// The TWO scalar fp16->fp32 reads (d_x, d_y) -- the sanctioned opaque piece.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The qh 5th-bit field read as TWO ALIGNED 16-bit halves (xb+2 / xb+4).
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// Per-block i32 scalar accumulator, reset each block.
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The INNER m1 strip loop (wide-LMUL default): vsetvl_e8m1 (VLMAX 16 at VLEN=128
// covers the whole 16-byte half-block in ONE strip, ggml's one-vwredsum anchor).
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// The weight is loaded UNSIGNED (vle8_v_u8m1); the q8 activations stay i8.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The 5-BIT decode chain (m1 width): the unsigned nibble unpack (vand 0x0F /
// vsrl 0x04), then the per-element 5th-bit injection from the broadcast qh halves
// -- the vid+c shift vector / vsrl_vv / vand 1 / vsll 4 / vncvt narrowing -- the
// vor into the nibble, the reinterpret + vsub 16 offset-binary bias, then the
// SAME signed vwmul/vwmacc product against the plain q8 halves.
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vid_v_u16m2"
// CHECK: call_opaque "__riscv_vadd_vx_u16m2"
// CHECK: call_opaque "__riscv_vmv_v_x_u16m2"
// CHECK: call_opaque "__riscv_vsrl_vv_u16m2"
// CHECK: call_opaque "__riscv_vand_vx_u16m2"
// CHECK: call_opaque "__riscv_vsll_vx_u16m2"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vsub_vx_i8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vsub_vx_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-block reduce into the i32 scalar (vwredsum + lane-0 extract), now i16m2.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The fp32 accumulate sumf = sumf + (d_x*d_y)*(float)sumi (ggml q5_0's scales-
// first order, quants.c:351), grouped into ONE emitc.expression -- IDENTICAL to
// the mf4 anchor fixture's fold (LMUL is the *how*, never the *what*).
// CHECK: %[[SUMI_RVAL:.*]] = load %[[SUMI]] : <!emitc.opaque<"int32_t">>
// CHECK: %[[SUMF_RVAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[ACCUM:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: %[[DXDY:.*]] = mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[FSUMI:.*]] = cast %[[SUMI_RVAL]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TERM:.*]] = mul %[[DXDY]], %[[FSUMI]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[NEXT:.*]] = add %[[SUMF_RVAL]], %[[TERM]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield %[[NEXT]] : !emitc.opaque<"float">
// CHECK: }
// CHECK: assign %[[ACCUM]] : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
