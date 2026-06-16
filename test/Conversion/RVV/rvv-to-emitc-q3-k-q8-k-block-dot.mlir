// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// q3_K (the 3-bit modern K-quant, the LAST common K-quant) -- the COMPLETE ggml
// ggml_vec_dot_q3_K_q8_K as STRUCTURED emitc IR (I5; ZERO raw() strings). The
// single typed op tcrv_rvv.q3_k_q8_k_block_dot COMPOSES mechanisms already in
// hand with three q3_K-specific twists: (1) the 2-bit qs unpack (q2_K's `(qs >>
// shift) & 3` over shifts {0,2,4,6}); (2) the SUBTRACTIVE hmask high-bit plane
// (q5_K's fixed 32-byte plane pattern but `-4` when the bit is UNSET -> SIGNED
// [-4,3]: `a = (low2 | (hbit<<2)) - 4`); (3) the q3_K-OWN SIGNED 6-bit scale
// dance (masks kmask1=0x03030303 / kmask2=0x0f0f0f0f, read SIGNED int8, then
// `scale_j = scales[j] - 32`). The fold is q6_K's NO-min deferred d.Sum(aux32)
// (symmetric -- NO min term, NO dmin). The fp32 ordering is byte-exact vs
// _generic; pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc29-q3_K/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q3_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] scratch, the
// uint32_t utmp[4] (the signed-scale dance scratch), the carried 8-lane fp32
// sums vector (the q6_K no-min deferred fold; NO scalar sumf accumulator).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CHECK: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The 2-bit + SUBTRACTIVE-hmask unpack: u8m2 load of the qs chunk, then for the
// low 2 bits vsrl/vand, for the SAME hmask plane vsrl/vand/vsll the high bit,
// OR them (vor_vv), u8->i8 reinterpret, then vsub 4 (the SUBTRACTIVE bias ->
// signed [-4,3]), vse8.
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vsll_vx_u8m2"
// CHECK: call_opaque "__riscv_vor_vv_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vsub_vx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The q3_K SIGNED 6-bit scale dance (STRUCTURED scalar emitc; NO raw()): three
// u32 word loads, then bitwise and/or/shift into utmp.
// CHECK: bitwise_and
// CHECK: bitwise_or
// CHECK: bitwise_left_shift
// CHECK: bitwise_right_shift
// The per-sub-block widening dot into the 8-lane aux32: the SIGNED scale
// `(int)sc[js] - 32` (a sub on the loaded int8), then i8mf2 loads -> vwmul ->
// vwmacc.vx i32m2.
// CHECK: %[[AUX32:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
// CHECK: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The NO-min DEFERRED fp32 fold (q6_K's): the fp16 read seam ONCE (NO dmin
// second read), vfcvt i32->f32, a SEPARATE vfmul then a SEPARATE vfadd (NEVER
// vfmacc/vfmadd), then the post-loop vse32 + SEQUENTIAL horizontal sum.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfmacc
// CHECK-NOT: call_opaque "__riscv_vfmadd
// CHECK-NOT: call_opaque "__riscv_vfredusum
// The post-loop SEQUENTIAL horizontal sum + the *s store, then return.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
