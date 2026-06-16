// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml IQ4_NL x Q8_0 block dot-product (tcrv_rvv.iq4_nl_q8_0_block_dot, the
// Family-A sibling that opens the CODEBOOK class) lowers to the COMPLETE structured
// kernel (I5; zero raw(), every value an emitc node): the outer block loop over
// nb = n / 32, the per-block address arithmetic (weight stride 18, activation stride
// 34) + dual fp16 scalar scale reads, the codebook-gather integer core, and the
// iq4_nl/q8_0 fp32 fold `sumf + (float)sumi * (d_x * d_y)` (scales-first).
// The genuinely-new codebook class mechanism: the 16-entry non-linear int8 table is
// emitted as a structured `static const int8_t[16]` decl, broadcast-loaded into a
// vreg ONCE above the block loop (vle8_v_i8m1), then each 4-bit nibble (split into
// the two UNSIGNED index lanes via vand 0x0F / vsrl 0x04) is mapped through the table
// by vrgather_vv_i8m1 -- NOT the q4_0 offset-binary vxor/vsll/vsra decode. The op
// pins the m1 anchor (the gather needs the table register's VLMAX >= 16). This pins
// the m1/elided shape -- EXACTLY the emission ssh-rvv-validated byte-exact vs ggml
// (artifacts/inc30-iq4_nl): one vsetvl_e8m1(16) half-block cover, NO inner strip loop.

module {
  tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq4_nl_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_nl_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>, integer_core_lmul = "m1", strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(
// The codebook is emitted as a structured static const int8_t[16] decl (the
// non-linear nibble->int8 lookup table kvalues_iq4nl[16]).
// CHECK: verbatim "static const int8_t tcrv_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the block loop), i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The outer block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-block address arithmetic: vx + ib*18, vy + ib*34.
// CHECK: mul %{{.*}}, %{{.*}}
// The two scalar fp16->fp32 reads.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The m1 half-block setvl + the packed weight u8 load + the two q8 signed halves.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The CODEBOOK class mechanism: split the nibble into the two UNSIGNED index lanes,
// then GATHER each through the broadcast table (NOT the q4_0 offset-binary decode).
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook, not nibble-8).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// The shared asymmetric widening product (i8m1 x i8m1 -> i16m2): low then + high.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// The carried-seed vwredsum into i32m1 + the scalar extract.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The iq4_nl fold: d_x * d_y FIRST, then (float)sumi * that, then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
