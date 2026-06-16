// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-22 — the COMPLETE ggml ggml_vec_dot_q5_1_q8_1 block kernel as STRUCTURED
// emitc IR (I5; ZERO raw() strings), the Family-B 5-bit sibling that COMBINES
// q5_0's 5-bit reconstruction and q4_1's scale+MIN fold. The single typed op
// tcrv_rvv.q5_1_q8_1_block_dot lowers to: an outer emitc.for block loop over
// nb = n / QK, per-block address arithmetic (vx + ib*24, vy + ib*36) via
// emitc.add/emitc.mul nodes, the FOUR scalar fp16->fp32 reads (d_x, d_y, m_x,
// s_y) plus the scalar qh-field read as two ALIGNED u16 halves (the sanctioned
// opaque pieces), an INNER mf4 strip loop running the 5-BIT UNSIGNED decode/
// product chain (the unsigned nibble unpack vand 0x0F / vsrl 0x04, the
// per-element 5th-bit injection from the broadcast qh halves via a vid+c shift
// vector / vsrl_vv / vand 1 / vsll 4 / vncvt, the vor into the nibble, the
// reinterpret WITHOUT the -16 bias) against the plain q8 halves (vwmul + vwmacc)
// + vwredsum into a per-block i32 scalar, the fp32 accumulate
// sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml's exact q5_1 statement), and the
// *s scalar store. Every emitted value is a NODE in the IR graph -- no
// emitc.verbatim with C control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q5_1_q8_1_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_1_q8_1_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 24 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 8 : i64, activation_quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 4 : i64, weight_min_byte_offset = 2 : i64, activation_sum_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer AoS block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-block address arithmetic (vx + ib*24, vy + ib*36) -- structured mul/add.
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg2, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// The FOUR scalar fp16->fp32 reads (d_x, d_y, m_x, s_y) -- the sanctioned opaque
// piece, four NODES.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The qh 5th-bit field read as TWO ALIGNED 16-bit halves (xb+4 / xb+6, NOT one
// misaligned uint32_t -- the two u16 halves are 2-aligned and value-identical to
// the LE (qh & 0xFFFF) / (qh >> 16) halves).
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// Per-block i32 scalar accumulator, reset each block.
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The INNER mf4 strip loop over the 16 weight bytes.
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// The weight is loaded UNSIGNED (vle8_v_u8mf4); the q8 activations stay i8.
// CHECK: call_opaque "__riscv_vle8_v_u8mf4"
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// The 5-BIT UNSIGNED decode chain: the unsigned nibble unpack (vand 0x0F / vsrl
// 0x04), then the per-element 5th-bit injection from the broadcast qh halves --
// the vid+c shift vector / vsrl_vv / vand 1 / vsll 4 / vncvt narrowing -- the vor
// into the nibble, then the value-identity reinterpret (NO vsub 16 bias -- q5_1's
// weight is an unsigned q5 in [0,31]), then the SAME signed vwmul/vwmacc product.
// CHECK: call_opaque "__riscv_vand_vx_u8mf4"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf4"
// CHECK: call_opaque "__riscv_vid_v_u16mf2"
// CHECK: call_opaque "__riscv_vadd_vx_u16mf2"
// CHECK: call_opaque "__riscv_vmv_v_x_u16mf2"
// CHECK: call_opaque "__riscv_vsrl_vv_u16mf2"
// CHECK: call_opaque "__riscv_vand_vx_u16mf2"
// CHECK: call_opaque "__riscv_vsll_vx_u16mf2"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf4"
// CHECK: call_opaque "__riscv_vor_vv_u8mf4"
// CHECK: call_opaque "__riscv_vor_vv_u8mf4"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf4_i8mf4"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf4_i8mf4"
// The q5_1 UNSIGNED path drops the offset-binary bias entirely -- NO vsub.
// CHECK-NOT: call_opaque "__riscv_vsub_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16mf2"
// Per-block reduce into the i32 scalar (vwredsum + lane-0 extract).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The fp32 accumulate sumf = sumf + ((d_x*d_y)*sumi + m_x*s_y) (ggml q5_1's exact
// statement, quants.c:394 -- IDENTICAL to q4_1's), grouped into ONE
// emitc.expression. The scale product (d_x*d_y)*sumi and the m_x*s_y MIN term are
// summed FIRST (the `+` binds before the `+=`), THEN added to sumf -- two summed
// products, the Family-B distinction. The add grouping is fp-significant; this
// exact tree is what makes it byte-exact.
// CHECK: %[[SUMI_RVAL:.*]] = load %[[SUMI]] : <!emitc.opaque<"int32_t">>
// CHECK: %[[SUMF_RVAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[ACCUM:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: %[[FSUMI:.*]] = cast %[[SUMI_RVAL]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[DXDY:.*]] = mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[SCALE:.*]] = mul %[[DXDY]], %[[FSUMI]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[MIN:.*]] = mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[AB:.*]] = add %[[SCALE]], %[[MIN]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[NEXT:.*]] = add %[[SUMF_RVAL]], %[[AB]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield %[[NEXT]] : !emitc.opaque<"float">
// CHECK: }
// CHECK: assign %[[ACCUM]] : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
