// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml MXFP4 x Q8_0 block dot-product (tcrv_rvv.mxfp4_q8_0_block_dot, the
// CODEBOOK-class sibling that opens the LAST structural quantization class: FP4
// (e2m1) weights with an E8M0 shared-exponent block scale) lowers to the COMPLETE
// structured kernel (I5; zero raw(), every value an emitc node): the outer block loop
// over nb = n / 32, the per-block address arithmetic (weight stride 17, activation
// stride 34), the STRUCTURED E8M0 -> fp32 half weight scale, the SINGLE q8_0 fp16
// scale read, the codebook-gather integer core, and the mxfp4/q8_0 fp32 fold
// `sumf + (float)sumi * (scale_x * d_y)` (scales-first).
// The genuinely-new FP4-class pieces: (1) the block format (block_mxfp4 = {uint8_t e;
// uint8_t qs[16]}, stride 17, the FP4 nibbles at +1 after the 1-byte E8M0 exponent);
// (2) the weight scale is NOT a fp16 read but the structured E8M0 reconstruction
// 2^(e-128) = a uint32_t `bits = (e<2) ? (0x00200000u << (e & 0x1F)) : ((e-1) << 23)`
// reinterpreted via `*(const float *)&bits`. The codebook is emitted as a structured
// `static const int8_t[16]` decl, broadcast-loaded ONCE (vle8_v_i8m1), then each 4-bit
// nibble (vand 0x0F / vsrl 0x04) is mapped through the table by vrgather_vv_i8m1. The
// op pins the m1/elided shape (the gather needs the table register's VLMAX >= 16) --
// EXACTLY the emission ssh-rvv-validated byte-exact vs ggml (artifacts/inc31-mxfp4).

module {
  tcrv.exec.kernel @ggml_vec_dot_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "m1", strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(
// The FP4 codebook is emitted as a structured static const int8_t[16] decl (the
// non-linear nibble->int8 lookup table kvalues_mxfp4[16] = 2*E2M1).
// CHECK: verbatim "static const int8_t tcrv_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the block loop), i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The outer block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-block address arithmetic: vx + ib*17, vy + ib*34.
// CHECK: mul %{{.*}}, %{{.*}}
// The STRUCTURED E8M0 -> fp32 half weight scale (the genuinely-new FP4-class piece):
// read the exponent byte, build the bit pattern via cmp / bitwise_and (mask) /
// bitwise_left_shift / sub / conditional, then reinterpret via the float pointer pun.
// CHECK: cmp lt
// CHECK: bitwise_and %{{.*}}, %{{.*}}
// CHECK: bitwise_left_shift %{{.*}}, %{{.*}}
// CHECK: bitwise_left_shift %{{.*}}, %{{.*}}
// CHECK: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"uint32_t">
// CHECK: apply "&"(%{{.*}}) : (!emitc.lvalue<!emitc.opaque<"uint32_t">>) -> !emitc.ptr<!emitc.opaque<"const uint32_t">>
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint32_t">> to !emitc.ptr<!emitc.opaque<"const float">>
// The SINGLE surviving q8_0 fp16->fp32 scale read (the weight fp16 read is GONE).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK-NOT: call_opaque "(float)*(const _Float16 *)"
// The m1 half-block setvl + the packed FP4 weight u8 load + the two q8 signed halves.
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
// The mxfp4 fold: scale_x * d_y FIRST, then (float)sumi * that, then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
