// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml IQ4_XS x Q8_K super-block dot-product (tcrv_rvv.iq4_xs_q8_k_block_dot,
// the CODEBOOK class's SUPER-BLOCK rung -- the super-block variant of iq4_nl) lowers
// to the COMPLETE structured kernel (I5; zero raw(), every value an emitc node): the
// outer super-block loop over nb = n / 256, the per-super-block address arithmetic
// (weight stride 136, activation stride 292), the d4d8 = fp16(x.d) * fp32(y.d) scale,
// the scales_h load, and a FLAT loop over 8 sub-blocks each doing the SIGNED 6-bit
// scale extraction (scalar emitc bitwise ops), the iq4_nl codebook-gather integer
// core, and the FLOAT-domain fold `sumf + d1 * (float)sumi` with d1 = d4d8*(ls-32).
// REUSED from iq4_nl: the 16-entry non-linear int8 table emitted as a structured
// `static const int8_t[16]` decl, broadcast-loaded ONCE (vle8_v_i8m1), each 4-bit
// nibble (vand 0x0F / vsrl 0x04) mapped through the table by vrgather_vv_i8m1, then
// the SAME asymmetric vwmul/vwmacc product + vwredsum. The op pins the m1 anchor
// (the gather needs the table register's VLMAX >= 16). NO min term (symmetric).

module {
  tcrv.exec.kernel @ggml_vec_dot_iq4_xs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_ggml_vec_dot_iq4_xs_q8_K(
// The codebook is emitted as a structured static const int8_t[16] decl (the SAME
// non-linear nibble->int8 lookup table kvalues_iq4nl[16] iq4_nl uses).
// CHECK: verbatim "static const int8_t tcrv_iq4_xs_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the super-block loop), i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The outer super-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-super-block address arithmetic: vx + ibl*136, vy + ibl*292.
// CHECK: mul %{{.*}}, %{{.*}}
// The fp16 weight d read + the fp32 activation d load -> d4d8 mul.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The SIGNED 6-bit scale extraction: scalar emitc bitwise shift/and/or (no string).
// CHECK: bitwise_right_shift
// CHECK: bitwise_and
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// d1 = d4d8 * (float)(ls - 32): a SEPARATE sub + cast + mul (NOT fused with sumi).
// CHECK: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CHECK: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The iq4_nl codebook integer core: m1 half-block setvl, weight u8 load, q8 halves.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The CODEBOOK mechanism: split the nibble into the two UNSIGNED index lanes, then
// GATHER each through the broadcast table (NOT a linear nibble-8 decode).
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// The shared asymmetric widening product + the seed-0 vwredsum + scalar extract.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The FLOAT-domain fold: ONE expression d1 * (float)sumi then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
