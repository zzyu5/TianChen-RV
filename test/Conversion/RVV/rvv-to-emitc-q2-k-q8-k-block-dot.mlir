// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// q2_K (the 2-bit modern K-quant) -- the COMPLETE ggml ggml_vec_dot_q2_K_q8_K
// as STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.q2_k_q8_k_block_dot REUSES the super-block scaffolding + the scale+min
// structure of q4_K but is SIMPLER in three ways: (1) the weights are 2-bit (4
// per qs byte), unpacked by `(qs >> shift) & 3` over the 4 shifts {0,2,4,6};
// (2) the per-sub-block scales/mins are SIMPLE 4-bit nibbles of the 16 direct
// `scales[16]` bytes (`sc[j] & 0xF` scale, `sc[j] >> 4` min -- NO 6-bit utmp/
// kmask bit-dance); (3) the positive fold is SCALAR -- a single per-super-block
// int isum, folded `sumf += dall*isum - dmin*summs` as ONE scalar fp32 statement
// (NO 8-lane sums vector, NO post-loop horizontal sum). The fp32 ordering is
// byte-exact vs _generic; pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc28-q2_K/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q2_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q2_K_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_q2_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] scratch.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// The scalar sumf, zeroed ONCE outside the super-block loop (NO 8-lane sums
// vector -- q2_K's positive term is the scalar isum).
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The 2-bit weight unpack: u8m2 load of the qs chunk, then vsrl/vand over the
// shifts {0,2,4,6}, u8->i8 reinterpret, vse8.
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// q2_K does NOT do the 6-bit utmp/kmask bit-dance (that is q4_K/q5_K). The 4-bit
// scale/min are simple nibbles of the direct scales[16] bytes.
// CHECK-NOT: bitwise_or
// The per-sub-block widening dot into the scalar isum: i8mf2 loads -> vwmul ->
// vwredsum -> vmv_x_s.
// CHECK: %[[ISUM:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CHECK: %[[SUMMS:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SCALAR fp32 fold: dy loaded once, dall/dmin via the fp16 read seam, then
// `sumf += dall*isum - dmin*summs` as ONE emitc.expression (two muls + a sub + an
// add). NO 8-lane vfcvt/vfmul/vfadd, NO post-loop vse32 horizontal sum.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: expression
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfmacc
// CHECK-NOT: call_opaque "__riscv_vfmadd
// CHECK-NOT: call_opaque "__riscv_vse32_v_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfredusum
// The *s store, then return.
// CHECK: return
