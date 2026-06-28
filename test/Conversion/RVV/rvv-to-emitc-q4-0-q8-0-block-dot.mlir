// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-2a — the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel as STRUCTURED
// emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.q4_0_q8_0_block_dot lowers to: an outer emitc.for block loop over
// nb = n / QK, per-block address arithmetic (vx + ib*18, vy + ib*34) via
// emitc.add/emitc.mul nodes, the two scalar fp16->fp32 reads via
// emitc.call_opaque "(float)*(const _Float16 *)" (the ONE sanctioned opaque
// piece, a typed IR node), an INNER mf4 strip loop reusing INC-1's offset-binary
// asymmetric i4xi8 decode/product chain (vxor.vx 0x88 -> vsll/vsra -> vwmul +
// vwmacc against the plain q8 halves) + vwredsum into a per-block i32 scalar, the
// left-associative fp32 accumulate sumf = sumf + ((float)sumi * d_x) * d_y
// (ggml's exact order), and the *s scalar store.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. Byte-exactness to ggml's real kernel is
// pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc2a-block-kernel-structured/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer AoS block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-block address arithmetic (vx + ib*18, vy + ib*34) -- structured mul/add.
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg5, %{{.*}}
// The two scalar fp16->fp32 reads -- the ONE sanctioned opaque piece, a NODE.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// Per-block i32 scalar accumulator, reset each block.
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The INNER mf4 strip loop over the 16 weight bytes.
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// The three i8/mf4 chunk loads: packed-i4 weight + plain q8 low + plain q8 high.
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: call_opaque "__riscv_vle8_v_i8mf4"
// INC-1's offset-binary asymmetric i4xi8 decode/product chain (reused).
// CHECK: call_opaque "__riscv_vxor_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16mf2"
// Per-block reduce into the i32 scalar (vwredsum + lane-0 extract).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The left-assoc fp32 accumulate sumf = sumf + ((float)sumi * d_x) * d_y,
// grouped into ONE emitc.expression so mlir-translate renders it as a SINGLE C
// statement that the compiler fuses into the SAME FMA ggml does (one rounding)
// under -ffp-contract=on/default -- so byte-exactness holds across all four
// -ffp-contract modes. The two scalar loads (sumi, sumf) stay OUTSIDE the
// expression (emitc.load lacks the CExpression trait); only the
// cast/mul/mul/add nodes form the expression's single def-use tree.
// CHECK: %[[SUMI_RVAL:.*]] = load %[[SUMI]] : <!emitc.opaque<"int32_t">>
// CHECK: %[[SUMF_RVAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[ACCUM:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: %[[FSUMI:.*]] = cast %[[SUMI_RVAL]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TDX:.*]] = mul %[[FSUMI]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[TERM:.*]] = mul %[[TDX]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[NEXT:.*]] = add %[[SUMF_RVAL]], %[[TERM]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield %[[NEXT]] : !emitc.opaque<"float">
// CHECK: }
// CHECK: assign %[[ACCUM]] : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
