// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// K2 (K-quant first super-block proof) -- the COMPLETE ggml
// ggml_vec_dot_q6_K_q8_K as STRUCTURED emitc IR (I5; ZERO raw() strings). The
// single typed op tcrv_rvv.q6_k_q8_k_block_dot reuses the K1 super-block integer
// core (the 6-bit ql+qh unpack into aux8[256], the nested sub-block int8-scaled
// i32 accumulation into the 8-lane aux32) and adds the DEFERRED two-level fp32
// fold + the fp32 *s store, producing the dot-product output byte-exact vs
// ggml's _generic. It lowers to: a function-scoped int8_t aux8[256] + float
// sums8[8] scratch, a vfloat32m2 `sums` accumulator zeroed ONCE outside the
// super-block loop, an outer emitc.for over nb = n / 256 (each iteration: the
// K1 integer core producing aux32, then `d = fp16(x.d)*y.d`, vfcvt i32->f32, a
// SEPARATE vfmul by d, a SEPARATE vfadd into sums), then a vse32 of sums into
// sums8 + a SEQUENTIAL 8-term scalar horizontal sum (NOT a vfredusum) into the
// fp32 *s.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. The fp32 ordering (deferred multi-lane fold
// via separate vfmul/vfadd + the sequential horizontal sum) is byte-exact vs
// _generic; pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc12-q6k-k2/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q6_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q6_K_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_q6_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q6_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q6_k_q8_k_block_dot", scale_model = "per-sub-block-int8-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, weight_qh_byte_offset = 128 : i64, weight_scales_byte_offset = 192 : i64, weight_d_byte_offset = 208 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] + float sums8[8].
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CHECK: subscript %[[AUX8]]
// CHECK: apply "&"
// CHECK: %[[SUMS8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"float">>
// The 8-lane fp32 sums accumulator, zeroed ONCE outside the super-block loop.
// CHECK: %[[SUMS:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The K1 integer core (unpack + sub-block i32 accumulation).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vor_vv_u8m2"
// CHECK: call_opaque "__riscv_vsub_vx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The deferred fp32 fold: fp16(x.d) read, y.d fp32 load, d = mul, vfcvt, then a
// SEPARATE vfmul then a SEPARATE vfadd (NEVER a fused vfmacc).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfmacc
// CHECK-NOT: call_opaque "__riscv_vfmadd
// The sequential horizontal sum: vse32 of the lanes into sums8, then 8 scalar
// add in ascending order (NOT a vfredusum), then the *s store.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfredusum
// CHECK-NOT: call_opaque "__riscv_vfredosum
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: return
