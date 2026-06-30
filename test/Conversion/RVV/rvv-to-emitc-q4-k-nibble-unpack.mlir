// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 1 -- the q4_K/q5_K Region-A plain 4-bit nibble unpack into
// the element-ordered int8_t aux8[256] scratch as a FIRST-CLASS generic op
// (tcrv_rvv.q4_k_nibble_unpack), auto-constructing the unpack instead of carrying
// it inside the monolithic q4_K integer core. ONE super-block: NO nb = n/256
// loop, NO 6-bit scale/min bit-dance, NO per-sub-block dot, NO fp32 fold (those
// are the deferred q4_K bricks). The op's lowering DECLARES the aux8[256] scratch
// and fills it.
//
// The emitted Region-A chain is BYTE-IDENTICAL to the monolithic q4_K integer
// core's Region A (rvv-to-emitc-q4-k-q8-k-aux-partial.mlir CHECK lines): the SAME
// emitQ4_KPlainNibbleUnpack helper emits both. Four 64-element chunks, each one
// literal-VL vsetvl_e8m2(32) + a 32-wide vle8 weight load, a vand 0x0F low nibble
// + a vsrl 4 high nibble, each reinterpreted u8m2 -> i8m2 and vse8-stored into
// aux8 (the q4_K nibbles are UNsigned [0,15], NO -32 bias). NO offset-binary
// xor-0x88 decode (that is the q4_0 sibling, not q4_K).

module {
  tcrv.exec.kernel @q4_k_nibble_unpack_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_nibble_unpack attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_nibble_unpack, sew = 32 : i64, source_kernel = "q4_k_nibble_unpack_kernel", status = "selected-lowering-boundary"} {
        %unpacked = tcrv_rvv.q4_k_nibble_unpack %vx, %vl {kind = "q4_k_nibble_unpack", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_qs_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_q4_k_nibble_unpack_kernel_q4_k_nibble_unpack(
// The op DECLARES the function-scoped int8_t aux8[256] scratch it fills.
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// The plain 4-bit nibble unpack: literal-VL vsetvl_e8m2(32) + vle8 weight load,
// vand 0x0F low nibble, reinterpret to i8, vse8 store into aux8, then vsrl 4 high
// nibble, reinterpret to i8, vse8 store (NO -32 bias, NO xor-0x88 offset-binary).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: subscript %[[AUX8]]
// CHECK: apply "&"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The q4_0 offset-binary decode chain must be ABSENT (this is a plain unpack).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8m2"
// CHECK: return
