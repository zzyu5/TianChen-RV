// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// q4_K K4a (the most-used modern K-quant) -- the ggml ggml_vec_dot_q4_K_q8_K
// INTEGER CORE as STRUCTURED emitc IR (I5; ZERO raw() strings, including the
// 6-bit scale/min bit-dance). The single typed op tcrv_rvv.q4_k_q8_k_aux_partial
// reproduces _generic's per-super-block INTEGER state right BEFORE the fp32
// d/dmin fold: aux32[8] (the per-sub-block uint6-scaled i32 accumulator) AND the
// 8 decoded 6-bit scales + 8 decoded 6-bit mins (the get_scale_min_k4 /
// utmp/kmask cross-byte decode). The fp32 fold + the min term + the ABI are K4b,
// out of scope here.
//
// It lowers to: an outer emitc.for super-block loop over nb = n / 256, a
// function-scoped int8_t aux8[256] scratch filled by the plain 4-bit nibble
// unpack (four 64-elem chunks, 32-wide e8m2 vand 0x0F low nibble + vsrl 4 high
// nibble + vse8 store, NO -32 bias), the STRUCTURED scalar bit-dance (three
// uint32 word loads, emitc.bitwise_and/_or/_left_shift/_right_shift into a
// 4-word utmp, a vse8 of the 16 [scales,mins] bytes through the scale/min
// output), a nested sub-block emitc.for over 8 sub-blocks applying the
// per-sub-block UINT6 scale in the i32 domain (vwmul i8xi8->i16 then vwmacc.vx
// i32 += scale*i16 into the 8-lane e32m2 aux32 accumulator), and a vse32 store
// of aux32[8] through the aux32 output pointer.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. Byte-exactness to ggml's _generic INTEGER
// state is pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc23-q4_K-k4a/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_K_q8_K_aux_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_K_q8_K_aux attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %a32 = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %sm = tcrv_rvv.runtime_abi_value {c_name = "scalemin", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_K_q8_K_aux, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_K_q8_K_aux_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_k_q8_k_aux_partial %vx, %vy, %a32, %sm, %n, %vl {kind = "ggml_q4_k_q8_k_aux_partial", scale_model = "per-sub-block-uint6-scale-i32-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_K_q8_K_aux_kernel_ggml_vec_dot_q4_K_q8_K_aux(
// The super-block count nb = n / 256 and the int8_t aux8[256] + uint32_t utmp[4].
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CHECK: subscript %[[AUX8]]
// CHECK: apply "&"
// CHECK: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-super-block address arithmetic (vx + ib*144, vy + ib*292).
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %{{.*}}, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %{{.*}}, %{{.*}}
// The 4-bit unpack: vle8 q4 load, vand 0x0F low nibble, vsrl 4 high nibble,
// reinterpret to i8, then vse8 store into aux8 (NO -32 bias).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The STRUCTURED scalar bit-dance: three uint32 word loads + bitwise ops.
// CHECK: bitwise_right_shift
// CHECK: bitwise_and
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// The 16 [scales,mins] decoded bytes stored through the scale/min output.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vse8_v_u8m1"
// The 8-lane aux32 accumulator, reset to zero per super-block.
// CHECK: %[[AUX32:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The nested sub-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The per-sub-block UINT6 scale (uint8 zero-extended), applied in the i32 domain.
// CHECK: subscript %{{.*}}[%{{.*}}]
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The aux32[8] store through the aux32 output pointer (NO fp32 fold -- K4b).
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// CHECK: return
