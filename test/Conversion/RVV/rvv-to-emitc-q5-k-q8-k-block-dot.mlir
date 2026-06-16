// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// q5_K (the 5-bit modern K-quant) -- the COMPLETE ggml
// ggml_vec_dot_q5_K_q8_K as STRUCTURED emitc IR (I5; ZERO raw() strings,
// including the 6-bit scale/min bit-dance AND the qh 5th-bit injection). The
// single typed op tcrv_rvv.q5_k_q8_k_block_dot REUSES q4_K's K4b machinery
// ENTIRELY -- the shared super-block integer core (the plain 4-bit nibble
// unpack into aux8[256], the STRUCTURED scalar 6-bit scale/min bit-dance via
// utmp/kmask, the nested sub-block uint6-scaled i32 accumulation into the 8-lane
// aux32, exposing the decoded scales/mins), the DEFERRED two-level fp32 fold,
// the q4_K MIN term, and the fp32 *s store -- with the ONLY new piece being the
// qh high-bit plane injection: per 32-element half h in 0..7, add 16 to the
// unpacked nibble for every element whose qh plane bit h is set (lifting q4 in
// [0,15] to q5 in [0,31]), mirroring _generic's `a[l] += (hm[l] & m ? 16 : 0)`.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. The qh injection is vand/vsrl/vsll/vadd in
// the u8 domain BEFORE the u8->i8 reinterpret. The fp32 ordering is byte-exact
// vs _generic; pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc27-q5_K/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q5_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q5_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_k_q8_k_block_dot", scale_model = "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 176 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_dmin_byte_offset = 2 : i64, weight_scales_byte_offset = 4 : i64, weight_qh_byte_offset = 16 : i64, weight_qs_byte_offset = 48 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] + uint32_t utmp[4] +
// float sums8[8].
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CHECK: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// CHECK: %[[SUMS8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"float">>
// The 8-lane fp32 sums accumulator + the scalar sumf, both zeroed ONCE outside
// the super-block loop.
// CHECK: %[[SUMS:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The shared integer core: 4-bit unpack of the qs nibble.
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// The qh 5th-bit plane is loaded (a SECOND u8m2 load, from a FIXED offset) and
// injected via vsrl/vand/vsll/vadd in the u8 domain (the ONLY q5_K-vs-q4_K node
// difference).
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vsll_vx_u8m2"
// CHECK: call_opaque "__riscv_vadd_vv_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The 6-bit scale/min bit-dance (scalar emitc.bitwise_*).
// CHECK: bitwise_right_shift
// CHECK: bitwise_and
// CHECK: bitwise_or
// q5_K (like K4b) does NOT store the 16 scale/min bytes (that is the K4a path).
// CHECK-NOT: call_opaque "__riscv_vse8_v_u8m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The MIN term: bsums (int16) multiplied by the decoded mins into sumi.
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// The deferred fp32 positive fold: SEPARATE vfmul then SEPARATE vfadd (NEVER a
// fused vfmacc).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// The MIN subtraction sumf -= dmin*sumi as ONE emitc.expression (a mul + a sub).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: expression
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK-NOT: call_opaque "__riscv_vfmacc
// CHECK-NOT: call_opaque "__riscv_vfmadd
// The sequential horizontal sum: vse32 of the lanes into sums8, then 8 scalar
// add (NOT a vfredusum), then the *s store.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfredusum
// CHECK-NOT: call_opaque "__riscv_vfredosum
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: return
