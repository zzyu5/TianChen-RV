// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// K1 (K-quant first super-block proof) -- the ggml ggml_vec_dot_q6_K_q8_K
// INTEGER CORE as STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed
// op tcrv_rvv.q6_k_q8_k_aux32_partial reproduces _generic's per-super-block
// aux32[8] integer state right BEFORE the fp32 d-multiply (the fp32 two-level
// fold is K2, out of scope here). It lowers to: an outer emitc.for super-block
// loop over nb = n / 256, a function-scoped int8_t aux8[256] scratch filled by
// the 6-bit ql+qh unpack (two 128-elem chunks, 4-strip 32-wide e8m2 vand/vsrl/
// vsll/vor/vsub reconstruct + vse8 store at the exact _generic element
// permutation), a nested sub-block emitc.for over 16 sub-blocks applying the
// per-sub-block int8 scale in the i32 domain (vwmul i8xi8->i16 then vwmacc.vx
// i32 += scale*i16 into the 8-lane e32m2 aux32 accumulator), and a vse32 store
// of aux32[8] through the output pointer.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. Byte-exactness to ggml's _generic aux32
// integer state is pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc11-q6k-k1/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q6_K_q8_K_aux32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q6_K_q8_K_aux32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q6_K_q8_K_aux32, sew = 32 : i64, source_kernel = "ggml_vec_dot_q6_K_q8_K_aux32_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q6_k_q8_k_aux32_partial %vx, %vy, %s, %n, %vl {kind = "ggml_q6_k_q8_k_aux32_partial", scale_model = "per-sub-block-int8-scale-i32-domain", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, weight_qh_byte_offset = 128 : i64, weight_scales_byte_offset = 192 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q6_K_q8_K_aux32_kernel_ggml_vec_dot_q6_K_q8_K_aux32(
// The super-block count nb = n / 256 and the int8_t aux8[256] scratch.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// The &aux8[0] base pointer for the contiguous sub-block reads.
// CHECK: subscript %[[AUX8]]
// CHECK: apply "&"
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-super-block address arithmetic (vx + ib*210, vy + ib*292).
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg2, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// The 6-bit unpack: vle8 ql/qh loads, vand/vsrl/vsll/vor reconstruct, reinterpret
// to i8, vsub 32 bias, then a vse8 store into aux8 (the element permutation).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vsll_vx_u8m2"
// CHECK: call_opaque "__riscv_vor_vv_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vsub_vx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The 8-lane aux32 accumulator, reset to zero per super-block.
// CHECK: %[[AUX32:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The nested sub-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The per-sub-block int8 scale, applied in the i32 domain.
// CHECK: subscript %{{.*}}[%{{.*}}]
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The aux32[8] store through the output pointer (NO fp32 fold -- K2).
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// CHECK: return
