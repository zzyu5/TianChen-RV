// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-9 — the COMPLETE ggml ggml_vec_dot_q4_1_q8_1 block kernel as STRUCTURED
// emitc IR (I5; ZERO raw() strings), the Family-B (scale+MIN, asymmetric) sibling
// of the Q4_0 lowering. The single typed op tcrv_rvv.q4_1_q8_1_block_dot lowers
// to: an outer emitc.for block loop over nb = n / QK, per-block address
// arithmetic (vx + ib*20, vy + ib*36) via emitc.add/emitc.mul nodes, the FOUR
// scalar fp16->fp32 reads (d_x, d_y, m_x, s_y) via emitc.call_opaque
// "(float)*(const _Float16 *)" (the ONE sanctioned opaque piece), an INNER mf4
// strip loop running the UNSIGNED-nibble asymmetric i4xi8 decode/product chain
// (vand 0x0F / vsrl 0x04 on the u8 weight lane -> reinterpret -> vwmul + vwmacc
// against the plain q8 halves) + vwredsum into a per-block i32 scalar, the fp32
// accumulate sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml's exact statement), and
// the *s scalar store. Every emitted value is a NODE in the IR graph -- no
// emitc.verbatim with C control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_1_q8_1_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_1_q8_1_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, weight_min_byte_offset = 2 : i64, activation_sum_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer AoS block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-block address arithmetic (vx + ib*20, vy + ib*36) -- structured mul/add.
// The q4_1 ABI has FOUR operands (n=%arg0, s=%arg1, vx=%arg2, vy=%arg3); no
// bs/bx/by/nrc strides in this typed body.
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg2, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// The FOUR scalar fp16->fp32 reads (d_x, d_y, m_x, s_y) -- the ONE sanctioned
// opaque piece, four NODES.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
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
// The UNSIGNED-nibble asymmetric i4xi8 decode/product chain (NO vxor / -8):
// vand 0x0F (low nibble) / vsrl 0x04 (high nibble, LOGICAL) on the u8 lane,
// value-identity reinterpret to i8, then the SAME signed vwmul/vwmacc.
// CHECK: call_opaque "__riscv_vand_vx_u8mf4"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf4"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf4_i8mf4"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf4_i8mf4"
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16mf2"
// Per-block reduce into the i32 scalar (vwredsum + lane-0 extract).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The fp32 accumulate sumf = sumf + ((d_x*d_y)*sumi + m_x*s_y) (ggml q4_1's exact
// statement, quants.c:319), grouped into ONE emitc.expression. The scale product
// (d_x*d_y)*sumi and the m_x*s_y MIN term are summed FIRST (the `+` binds before
// the `+=`), THEN added to sumf -- two summed products, the Family-B distinction
// from q4_0's single block term. The add grouping is fp-significant
// (`(sumf+A)+B != sumf+(A+B)`); this exact tree is what makes it byte-exact.
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
