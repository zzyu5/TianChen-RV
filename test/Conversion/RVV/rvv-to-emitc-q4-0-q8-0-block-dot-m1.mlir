// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-4 — the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel, integer core
// re-anchored at LMUL m1 (the resource-aware Gearbox/N3 LMUL choice on a real
// llama.cpp kernel). The single typed op tcrv_rvv.q4_0_q8_0_block_dot carries
// integer_core_lmul = "m1" -- a bounded resource/scheduling fact (the *how*, not
// the *what*) -- so the per-block integer dot anchors at i8m1 -> i16m2 -> a
// SINGLE vwredsum (matching ggml's hand-written reduction anchor: the whole
// 16-byte half-block in ONE strip at VLEN=128, one vwredsum per block, no inner
// 4-chunk loop), instead of the INC-2a default mf4 -> mf2 -> i32m1 with a
// 4-chunk inner strip. The dot product stays BYTE-EXACT (vwredsum sums the same
// integer set; integer add is order-independent; the fp32 accumulate is
// unchanged). The strip loop is retained for VLEN robustness: at VLEN=128 it runs
// once; a VLEN < 128 board re-strips correctly via the sumi-carrying seed.
//
// Byte-exactness vs ggml's real kernel + _generic is pinned by the ssh-rvv
// artifact under .trellis/tasks/.../artifacts/inc4-m1-perf/.

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
        %dot = tcrv_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
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
// The strip loop re-anchored at m1: vsetvl_e8m1 (VLMAX 16 at VLEN=128 -> ONE
// strip per block, the win), retained for VLEN robustness.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// The three i8/m1 chunk loads: packed-i4 weight + plain q8 low + plain q8 high.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// INC-1's offset-binary asymmetric i4xi8 decode/product chain, re-anchored at m1.
// CHECK: call_opaque "__riscv_vxor_vx_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// The widening product at i16m2 (the wider LMUL the m1 source drives).
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-block reduce: the i16m2 product reduces into i32m1 lane 0 in ONE vwredsum
// (vwredsum is LMUL-independent in result -- the same integer set the mf4 path
// reduces over 4 strips, here in one).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The left-assoc fp32 accumulate sumf = sumf + ((float)sumi * d_x) * d_y,
// UNCHANGED from the mf4 path -- grouped into ONE emitc.expression that fuses
// into the SAME FMA ggml does under -ffp-contract=on/default.
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
