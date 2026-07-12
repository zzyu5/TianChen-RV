// END-TO-END production-export CLOSURE for the C3 (N=3 offset-binary) front door:
// the front door's OWN auto-constructed nibble integer-core body now flows through
// the FULL production-export pipeline (--weft-materialize-emission-plans) and emits
// correct EmitC. This is the P1e W5 payoff -- the FIRST N=3 descriptor-driven route
// proven end-to-end, validating the N-operand descriptor refactor: three input
// buffers (packed-i4 weight + the two plain-i8 q8 activation halves) route through
// a SINGLE generic-op composition with ZERO new emit vocabulary and ZERO new
// special-case route (the arity is derived from the contraction descriptor).
//
// WHY this is a distinct, load-bearing test (vs the front-door fixture
// test/Transforms/RVV/rvv-packed-i4-offset-binary-dot-source-front-door.mlir): that
// fixture runs front-door --> --weft-rvv-lower-to-emitc DIRECTLY. This test inserts
// --weft-materialize-emission-plans in the MIDDLE -- the production-export chain --
// proving the N=3 route survives emission-plan materialization end-to-end, not just
// the direct lower. The C3 op is ALREADY a first-class typed op that lowers to
// EmitC; every P1e wall (W1-W4) was an un-migrated consumer of the 3-operand shape,
// and W5 locks the culminating end-to-end emission as durable regression coverage.
//
// THE CHAIN (matches the front-door CORE anchor): three i8mf4 loads (w / qlo / qhi)
//   --> offset-binary decode (vxor 0x88 the WEIGHT --> vsll/vsra low + vsra high
//       nibble sign-extend)
//   --> asymmetric widening product: LOW nibble x qlo (vwmul) then HIGH nibble x qhi
//       (vwmacc) -- the low->qlo / high->qhi pairing is SSA-pinned below so a swap
//       fails the test
//   --> signed widening reduce (vwredsum i16mf2 -> i32m1)
//   --> vse32 store.
//
// CAPABILITY -- NO VLEN FLIP (contrast with the dequant e2e sibling). q4_0's nibble
// half-block integer core is pinned at i8mf4-i16mf2-i32m1 (strip vsetvl e32m1) at
// EVERY Zvl128b tier (integer-core anchor "i8mf4-i16mf2-i32m1-no-vlen-flip"). Unlike
// the dequant front door -- whose realized strip capability-FLIPS m2/m4 (VLEN128) ->
// m1/m2 (VLEN256) -- the offset-binary integer core is VLEN-INVARIANT: the VLEN256
// export (march=rv64gcv_zvl256b) is BYTE-IDENTICAL to VLEN128. The EMITC256 anchors
// below therefore mirror EMITC (intended, not redundant) and add EMITC256-NOT pins
// so any accidental future strip flip (m1/m2/m4 forms) fails the test.
//
// HOST-ONLY: correctness of the nibble decode + asymmetric product + widening reduce
// is covered by the existing host scalar oracle; this asserts only that the C3
// front-door output now EXPORTS (exit 0) through the production plan chain with the
// correct 6-arg signature + intrinsics. NO board / NO perf claim.

// VLEN128 production-export: front door auto-constructs the body, materializes the
// emission plan, lowers to EmitC.
// RUN: weft-opt %s --weft-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// VLEN256 production-export: the SAME generic source, capability tier rv64gcv_zvl256b.
// The offset-binary integer core does NOT flip -- byte-identical to VLEN128.
// RUN: weft-opt %s --weft-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv_zvl256b --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC256

// ===================== VLEN128 EMITTED C3 (N=3) CHAIN ========================
// The 6-arg EmitC signature: w / qlo / qhi = const int8_t*, acc = const int32_t*,
// out = int32_t*, n = size_t. This is the N=3 descriptor projection (three input
// buffers), NOT the legacy N=2 lhs/rhs pair.
// EMITC: emitc.func @weft_emitc_rvv_packed_i4_offset_binary_dot_i8_from_source_rvv_packed_i4_offset_binary_dot_i8(
// EMITC-SAME: %arg0: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC-SAME: %arg1: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC-SAME: %arg2: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC-SAME: %arg3: !emitc.ptr<!emitc.opaque<"const int32_t">>,
// EMITC-SAME: %arg4: !emitc.ptr<!emitc.opaque<"int32_t">>,
// EMITC-SAME: %arg5: !emitc.opaque<"size_t">)
// The strip vsetvl is e32m1 (SEW=32, LMUL=m1) -- the pinned no-flip integer core.
// EMITC: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop i32 seed: out[0] = acc[0].
// EMITC: %[[ACCSCALAR:.*]] = load
// EMITC: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// EMITC: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// EMITC: for %{{.*}} = %{{.*}} to %{{.*}} step
// EMITC: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The three i8/mf4 source loads: w (slot0) / qlo (slot1) / qhi (slot2).
// EMITC: %[[W:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// EMITC: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// EMITC: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// Offset-binary -> two's-complement: xor 0x88 the WEIGHT (w = %[[W]]) only.
// EMITC: %[[WXOR:.*]] = call_opaque "__riscv_vxor_vx_i8mf4"(%[[W]],
// Low nibble: shift into the high nibble then arithmetic-shift back (sign-extend).
// EMITC: %[[WLOSH:.*]] = call_opaque "__riscv_vsll_vx_i8mf4"(%[[WXOR]],
// EMITC: %[[V0:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WLOSH]],
// High nibble: arithmetic-shift sign-extends it in place.
// EMITC: %[[V1:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WXOR]],
// PAIRING PIN: LOW nibble (%[[V0]]) x qlo (%[[QLO]]) via vwmul; HIGH nibble (%[[V1]])
// x qhi (%[[QHI]]) via vwmacc. A low<->high or qlo<->qhi swap fails these captures.
// EMITC: call_opaque "__riscv_vwmul_vv_i16mf2"(%[[V0]], %[[QLO]], %{{.*}}) : (!emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// EMITC: call_opaque "__riscv_vwmacc_vv_i16mf2"(%{{.*}}, %[[V1]], %[[QHI]], %{{.*}}) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// In-loop i32 running seed + the signed widening reduce (vwredsum).
// EMITC: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// EMITC: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"({{.*}}, %[[SEED]], %{{.*}}) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// EMITC: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// EMITC: return

// ===================== VLEN256 EMITTED C3 (N=3) CHAIN -- NO FLIP =============
// Byte-identical to VLEN128 (integer core VLEN-invariant). Same 6-arg signature,
// same pinned mf4/mf2/m1 strip, same low->qlo / high->qhi pairing. The EMITC256-NOT
// lines pin the NO-FLIP: no accidental m1/m2/m4 strip may appear.
// EMITC256: emitc.func @weft_emitc_rvv_packed_i4_offset_binary_dot_i8_from_source_rvv_packed_i4_offset_binary_dot_i8(
// EMITC256-SAME: %arg0: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC256-SAME: %arg1: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC256-SAME: %arg2: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC256-SAME: %arg3: !emitc.ptr<!emitc.opaque<"const int32_t">>,
// EMITC256-SAME: %arg4: !emitc.ptr<!emitc.opaque<"int32_t">>,
// EMITC256-SAME: %arg5: !emitc.opaque<"size_t">)
// The strip stays e32m1 -- NO flip to e8m1/e8m2.
// EMITC256: call_opaque "__riscv_vsetvl_e32m1"
// EMITC256-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMITC256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// The three loads stay i8mf4 -- NO flip to i8m1/i8m2.
// EMITC256: %[[W256:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_i8m1"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_i8m2"
// EMITC256: %[[QLO256:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// EMITC256: %[[QHI256:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// Same offset-binary decode.
// EMITC256: %[[WXOR256:.*]] = call_opaque "__riscv_vxor_vx_i8mf4"(%[[W256]],
// EMITC256: %[[WLOSH256:.*]] = call_opaque "__riscv_vsll_vx_i8mf4"(%[[WXOR256]],
// EMITC256: %[[V0256:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WLOSH256]],
// EMITC256: %[[V1256:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WXOR256]],
// PAIRING PIN (VLEN256): LOW x qlo, HIGH x qhi -- product stays i16mf2, NO flip.
// EMITC256: call_opaque "__riscv_vwmul_vv_i16mf2"(%[[V0256]], %[[QLO256]], %{{.*}})
// EMITC256-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMITC256-NOT: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC256: call_opaque "__riscv_vwmacc_vv_i16mf2"(%{{.*}}, %[[V1256]], %[[QHI256]], %{{.*}})
// EMITC256-NOT: call_opaque "__riscv_vwmacc_vv_i16m2"
// EMITC256-NOT: call_opaque "__riscv_vwmacc_vv_i16m4"
// The reduce stays i16mf2 -> i32m1 -- NO flip to i16m2/i16m4 source.
// EMITC256: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC256: return

module attributes {weft_rvv.source_front_door = "bounded_packed_i4_offset_binary_dot_source"} {
  func.func @source_packed_i4_dot(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>, %acc: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}
