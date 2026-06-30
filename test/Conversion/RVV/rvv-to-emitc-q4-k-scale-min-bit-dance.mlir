// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 2 -- the q4_K/q5_K Region-B 6-bit scale/min bit-dance (the
// get_scale_min_k4 / utmp/kmask cross-byte decode into the 16 [scales,mins]
// bytes) as a FIRST-CLASS generic op (tcrv_rvv.q4_k_scale_min_bit_dance),
// auto-constructing the bit-dance instead of carrying it inside the monolithic
// q4_K integer core. ONE super-block: NO nb = n/256 loop, NO plain 4-bit nibble
// unpack, NO per-sub-block dot, NO MIN term, NO fp32 fold (those are BRICK 1 /
// the deferred q4_K bricks). The op's lowering DECLARES the uint32_t utmp[4]
// scratch and fills it, then stores the 16 decoded bytes through a local
// uint8_t scale_min[16] sink as the observable.
//
// The emitted Region-B bit-dance chain is BYTE-IDENTICAL to the monolithic q4_K
// integer core's Region B (rvv-to-emitc-q4-k-q8-k-aux-partial.mlir CHECK lines):
// the SAME emitQ4_KScaleMinBitDanceCore helper emits both. Three (const uint32_t
// *)(weight_base + 4) word loads, the kmask1/2/3 (0x3f3f3f3f / 0x0f0f0f0f /
// 0x03030303) cross-byte shuffle via STRUCTURED scalar emitc.bitwise ops into
// utmp[0..3], scales = (const uint8_t *)&utmp[0] (NO raw string blob).

module {
  tcrv.exec.kernel @q4_k_scale_min_bit_dance_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_scale_min_bit_dance attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_scale_min_bit_dance, sew = 32 : i64, source_kernel = "q4_k_scale_min_bit_dance_kernel", status = "selected-lowering-boundary"} {
        %decoded = tcrv_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_q4_k_scale_min_bit_dance_kernel_q4_k_scale_min_bit_dance(
// The op DECLARES the function-scoped uint32_t utmp[4] scratch it fills.
// CHECK: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// The STRUCTURED scalar bit-dance: pin the EXACT kmask constants (kmask1 =
// 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303) emitted as emitc.literal,
// plus the shift amounts 4 and 6.
// CHECK: %[[KMASK1:.*]] = literal "0x3f3f3f3f" : !emitc.opaque<"uint32_t">
// CHECK: %[[KMASK2:.*]] = literal "0x0f0f0f0f" : !emitc.opaque<"uint32_t">
// CHECK: %[[KMASK3:.*]] = literal "0x03030303" : !emitc.opaque<"uint32_t">
// CHECK: %[[SH4:.*]] = literal "4" : !emitc.opaque<"uint32_t">
// CHECK: %[[SH6:.*]] = literal "6" : !emitc.opaque<"uint32_t">
// The three packed uint32 scale words at (const uint32_t *)(weight_base + 4):
// add the scales offset, cast to const uint32_t *, then subscript[0/1/2] + load.
// CHECK: %[[SCOFF:.*]] = literal "4" : !emitc.opaque<"size_t">
// CHECK: add %arg1, %[[SCOFF]]
// CHECK: %[[SCPTR:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const uint32_t">>
// CHECK: subscript %[[SCPTR]]
// CHECK: load
// CHECK: %[[W0:.*]] = cast %{{.*}} : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
// CHECK: subscript %[[SCPTR]]
// CHECK: %[[W1:.*]] = cast %{{.*}} : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
// CHECK: subscript %[[SCPTR]]
// CHECK: %[[W2:.*]] = cast %{{.*}} : !emitc.opaque<"const uint32_t"> to !emitc.opaque<"uint32_t">
// u3 = ((w2 >> 4) & kmask2) | (((w1 >> 6) & kmask3) << 4)  (inner (w1>>6) term
// emitted first, then the (w2>>4) term, then the or -- the shared-helper order).
// CHECK: %[[T0:.*]] = bitwise_right_shift %[[W1]], %[[SH6]]
// CHECK: bitwise_and %[[T0]], %[[KMASK3]]
// CHECK: bitwise_left_shift
// CHECK: %[[T1:.*]] = bitwise_right_shift %[[W2]], %[[SH4]]
// CHECK: bitwise_and %[[T1]], %[[KMASK2]]
// CHECK: %[[U3:.*]] = bitwise_or
// u2 = w1 & kmask1
// CHECK: %[[U2:.*]] = bitwise_and %[[W1]], %[[KMASK1]]
// u1 = (w2 & kmask2) | (((w0 >> 6) & kmask3) << 4)
// CHECK: %[[T2:.*]] = bitwise_right_shift %[[W0]], %[[SH6]]
// CHECK: bitwise_and %[[T2]], %[[KMASK3]]
// CHECK: bitwise_left_shift
// CHECK: bitwise_and %[[W2]], %[[KMASK2]]
// CHECK: %[[U1:.*]] = bitwise_or
// u0 = w0 & kmask1
// CHECK: %[[U0:.*]] = bitwise_and %[[W0]], %[[KMASK1]]
// The 4-word utmp[0..3] store (assign, in u0/u1/u2/u3 index order) + scales =
// (const uint8_t *)&utmp[0].
// CHECK: subscript %[[UTMP]]
// CHECK: assign %[[U0]]
// CHECK: subscript %[[UTMP]]
// CHECK: assign %[[U1]]
// CHECK: subscript %[[UTMP]]
// CHECK: assign %[[U2]]
// CHECK: subscript %[[UTMP]]
// CHECK: assign %[[U3]]
// CHECK: subscript %[[UTMP]]
// CHECK: apply "&"
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint32_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
// The store_scale_min observable: vse8 the 16 decoded [scales,mins] bytes through
// a local uint8_t scale_min[16] sink.
// CHECK: %[[SCALEMIN:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<16x!emitc.opaque<"uint8_t">>
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vse8_v_u8m1"
// The plain 4-bit nibble unpack chain must be ABSENT (this is the bit-dance, NOT
// BRICK 1's Region-A nibble unpack).
// CHECK-NOT: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: return
