// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml Q1_0 x Q8_0 block dot-product (tcrv_rvv.q1_0_q8_0_block_dot, the BINARY
// {-1,+1} class -- one of the last three uncommon ggml dot kernels) lowers to the
// COMPLETE structured kernel (I5; zero raw(), every value an emitc node): the outer
// super-block loop over nb = n / 128, the four UNROLLED q8_0 sub-blocks (each with
// its own d1_k fp16 scale + ONE 32-lane sign-decode body), the BINARY sign decode,
// and the two-level fp32 fold `sumf += d0 * sum_k(d1_k * sumi_block_k)`.
// The genuinely-new BINARY-class mechanism: each q1_0 weight bit is a SIGN (set ->
// +q8, clear -> -q8) and the q8 value is the magnitude. The emitter runs ggml's
// SHIPPED _vl128 lane structure: the 4 packed bit-bytes load DIRECTLY into the i8
// sign mask via vlm_v_b{ratio} (the bits ARE the mask), the 32 q8 quants load via
// vle8, the sign is applied IN THE i8 DOMAIN via vmerge(vneg_i8(q8), q8, mask), and
// ONE vwredsum widens i8 -> i16m1 per sub-block. The anchor MOVES with VLEN (the
// 32-element sub-block straddles m1's i8 VLMAX boundary between 128/256): at VLEN128
// integer_core_lmul = "m2" (e8m1 VLMAX 16 < 32) so vlm_v_b4; at VLEN256 m1 spans it.
// The 0.033x LOSS root-cause (the prior 8-lane / 4-vwredsum-per-sub-block form) is
// closed by collapsing to ONE 32-lane reduce. byte-exact vs ggml's _vl128.

module {
  tcrv.exec.kernel @ggml_vec_dot_q1_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q1_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q1_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q1_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q1_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q1_0_q8_0_block_dot", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m2", minimum_vlen = 128 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0(
// The function-scoped fp32 accumulator + the SUPER-block count nb = n / 128.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The kmask table is GONE (the packed bits ARE the mask via vlm_v_b{ratio}).
// CHECK-NOT: tcrv_q1_0_kmask
// CHECK-NOT: call_opaque "__riscv_vmv_v_x_u8m
// CHECK-NOT: call_opaque "__riscv_vand_vv_u8m
// CHECK-NOT: call_opaque "__riscv_vmsne_vx_u8m
// CHECK-NOT: call_opaque "__riscv_vwcvt_x_x_v_i16
// The outer super-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-super-block weight address (vx + ib*18) + the d0 fp16 read.
// CHECK: mul %{{.*}}, %{{.*}}
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The per-super-block float accumulator (sumi) RESET to 0.0f.
// CHECK: literal "0.0f"
// The q8 sub-block address (vy + (ib*4 + k)*34) + the d1_k fp16 read.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The 32-lane sub-block setvl at the m2 anchor (e8m2 VLMAX 32 spans the block).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// The BINARY sign decode: vlm_v_b4 the 4 packed bit-bytes DIRECTLY into the i8 mask.
// CHECK: call_opaque "__riscv_vlm_v_b4"
// The 32 q8 quants, then negate + merge IN THE i8 DOMAIN (ggml's _vl128 exact ops).
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vneg_v_i8m2"
// CHECK: call_opaque "__riscv_vmerge_vvm_i8m2"
// The q4_0 offset-binary nibble decode must be ABSENT (this is a binary sign plane).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// ONE vwredsum per sub-block, widening i8m2 -> i16m1 (seeded by zero) + extract.
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i8m2_i16m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i16m1_i16"
// The per-sub-block fold: sumi = sumi + d1 * (float)sumi_block (ONE emitc.expression).
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The super-block fold: sumf = sumf + d0 * sumi.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
