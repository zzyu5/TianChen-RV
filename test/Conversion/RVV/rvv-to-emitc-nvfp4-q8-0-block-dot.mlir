// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml NVFP4 x Q8_0 block dot-product (tcrv_rvv.nvfp4_q8_0_block_dot, the SECOND
// FP4-class sibling: NVIDIA's FP4 -- the SAME e2m1 kvalues_mxfp4 codebook as mxfp4, but
// a per-SUB-block UE4M3 fp8 scale on a QK=64 SUPER-block) lowers to the COMPLETE
// structured kernel (I5; zero raw(), every value an emitc node): the outer super-block
// loop over nb = n / 64, the four UNROLLED 16-element sub-blocks (each with its own
// UE4M3 scale + q8 block/half selection + 8-lane strip), the codebook-gather integer
// core, and the nvfp4/q8_0 fp32 fold `sumf + (dy * d) * (float)sumi`.
// The genuinely-new NVFP4-class pieces: (1) the super-block format (block_nvfp4 =
// {uint8_t d[4]; uint8_t qs[32]}, stride 36, four UE4M3 scales then the FP4 nibbles at
// +4, spanning TWO block_q8_0 blocks); (2) the weight scale is NOT mxfp4's E8M0 bit
// dance but the structured UE4M3 -> fp32 decode (exp/man split, the two ldexpf branches,
// *0.5f, the two specials e==0/e==0x7F -> 0.0f via logical_or). The codebook is emitted
// as a structured `static const int8_t[16]` decl, broadcast-loaded ONCE (vle8_v_i8m1),
// then each 4-bit nibble (vand 0x0F / vsrl 0x04) is mapped through the table by
// vrgather_vv_i8m1. The op pins the m1 shape (the gather needs the table register's
// VLMAX >= 16) -- EXACTLY the emission ssh-rvv-validated byte-exact vs ggml's
// _generic (= the real board kernel; artifacts/inc40-nvfp4).

module {
  tcrv.exec.kernel @ggml_vec_dot_nvfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_nvfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_nvfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_nvfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0(
// The FP4 codebook is emitted as a structured static const int8_t[16] decl -- REUSED
// from mxfp4 (the SAME kvalues_mxfp4 = 2*E2M1 table).
// CHECK: verbatim "static const int8_t tcrv_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The function-scoped fp32 accumulator + the SUPER-block count nb = n / 64.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the block loop), i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The outer super-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-super-block weight address (vx + ib*36) + the q8 block-pair base (2*ib).
// CHECK: mul %{{.*}}, %{{.*}}
// The STRUCTURED UE4M3 -> fp32 weight scale (the genuinely-new NVFP4-class piece): the
// exp/man split (bitwise_right_shift / bitwise_and), the two ldexpf branches, the
// exp==0 conditional, the *0.5f, then the two specials (e==0 || e==0x7F -> 0.0f).
// CHECK: bitwise_right_shift %{{.*}}, %{{.*}}
// CHECK: bitwise_and %{{.*}}, %{{.*}}
// CHECK: bitwise_and %{{.*}}, %{{.*}}
// CHECK: call_opaque "ldexpf"
// CHECK: call_opaque "ldexpf"
// CHECK: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// CHECK: logical_or %{{.*}}, %{{.*}}
// CHECK: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// mxfp4's E8M0 bit-dance MUST be absent (this is the UE4M3 ldexpf path, not a misroute).
// CHECK-NOT: 0x00200000
// The q8_0 fp16->fp32 scale read (the OTHER sanctioned opaque scalar piece).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The 8-lane sub-block setvl + the packed FP4 weight u8 load + the two q8 signed halves.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The CODEBOOK class mechanism: split the nibble into the two UNSIGNED index lanes,
// then GATHER each through the broadcast table.
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook, not nibble-8).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// The shared asymmetric widening product (i8m1 x i8m1 -> i16m2): low then + high.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// The vwredsum into i32m1 + the scalar extract.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The nvfp4 fold: dy * d FIRST, then * (float)sumi, then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
