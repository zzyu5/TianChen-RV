// Track B G1, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the q4_0 nibble
// INTEGER-CORE body from a marked GENERIC nibble-core source, instead of routing a
// monolithic op to a per-kernel hand emitter. The auto-constructed body is the
// single-strip generic-op composition
//   load x3 (packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> tcrv_rvv.packed_i4_offset_binary_x_i8_product (offset-binary nibble decode
//        + asymmetric widening product -- ALREADY a first-class generic op)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> tcrv_rvv.store
// and it lowers BYTE-IDENTICAL (modulo kernel symbol name) to the existing
// hand-authored nibble-core lit
// test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir,
// proving auto-construction REACHES nibble-decode: the nibble unpack needs NO new
// emitter vocabulary; it is already a typed op the front door constructs.
//
// HONEST SCOPE -- the nibble integer CORE ONLY (NOT the full q4_0 KERNEL). There is
// NO nb = n / QK outer block loop, NO per-block dual fp16 scale read, and NO
// left-associative fp32 fold (those need NEW generic ODS vocabulary -- full G1,
// DEFERRED). The monolithic tcrv_rvv.q4_0_q8_0_block_dot op, its KERNEL front door,
// and the hand emitter all STAY.
//
// CAPABILITY -- NO FLIP claimed. q4_0's nibble half-block integer core is pinned at
// i8mf4-i16mf2-i32m1 (strip vsetvl e32m1) at every Zvl128b tier; there is NO
// VLEN128-vs-VLEN256 byte-flip here. The shared schedule authority is consulted as
// the LEGALITY GATE only (fail-closed if the integer-core path is pruned).

// The auto-constructed nibble integer-core body (the materialized kernel scaffold).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=BODY
//
// The auto-constructed body lowered to EmitC: the SAME offset-binary nibble decode
// + asymmetric product + widening-reduce intrinsic chain the existing nibble-core
// lit pins (vxor 0x88 -> vsll/vsra sign-extend -> vwmul/vwmacc -> vwredsum), proving
// auto-construction reaches nibble-decode byte-for-byte.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=CORE
//
// FAIL-CLOSED (I7): a marker-carrying source with the WRONG (4-arg, q4_0 KERNEL-
// shaped) signature -- not the 6-role nibble integer-core operator identity -- is
// REJECTED, not silently lowered.
// RUN: not tcrv-opt %S/Inputs/packed-i4-offset-binary-dot-source-wrong-signature.mlir --tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv 2>&1 | FileCheck %s --check-prefix=WRONGSIG

module attributes {tcrv_rvv.source_front_door = "bounded_packed_i4_offset_binary_dot_source"} {
  func.func @source_packed_i4_dot(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>, %acc: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED tcrv_rvv NIBBLE-CORE BODY ============
// The generic nibble-core source becomes a tcrv.exec.kernel with the auto-built
// load x3 / packed_i4_offset_binary_x_i8_product / standalone_reduce / store body.
// NO per-kernel emitter authored this body.
// BODY: tcrv.exec.kernel @rvv_packed_i4_offset_binary_dot_i8_from_source
// BODY: tcrv.exec.variant @rvv_packed_i4_offset_binary_dot_i8
// The pinned no-flip nibble integer-core anchor (audit-only provenance).
// BODY-SAME: tcrv_rvv.packed_i4_integer_core_anchor = "i8mf4-i16mf2-i32m1-no-vlen-flip"
// The strip vsetvl the mf4 anchor issues is e32m1 (SEW=32, LMUL=m1).
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The three i8/mf4 source loads: packed-i4 weight + plain q8 low + plain q8 high.
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "mf4">
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "mf4">
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "mf4">
// The auto-constructed offset-binary nibble decode + asymmetric widening product.
// BODY: tcrv_rvv.packed_i4_offset_binary_x_i8_product
// BODY-SAME: kind = "signed_packed_i4_offset_binary_x_i8_product"
// BODY-SAME: product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"
// BODY-SAME: -> !tcrv_rvv.vector<i16, "mf2">
// BODY: tcrv_rvv.standalone_reduce
// BODY-SAME: kind = "signed_widening_reduce_add"
// BODY-SAME: -> !tcrv_rvv.vector<i32, "m1">
// BODY: tcrv_rvv.store
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_packed_i4_offset_binary_dot_i8_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_packed_i4_offset_binary_dot_i8
// BODY: tcrv.exec.fallback @rvv_packed_i4_offset_binary_dot_i8_scalar_fallback

// ===================== EMITTED NIBBLE-CORE INTRINSIC CHAIN ===================
// The auto-constructed body lowers to the exact nibble integer-core chain the
// existing hand-authored lit pins: vsetvl_e32m1, the i32 acc seed, the three
// i8mf4 loads, the offset-binary decode (xor 0x88 -> low/high sign-extend), the
// asymmetric vwmul/vwmacc product, and the vwredsum widening reduce.
// CORE: emitc.func @tcrv_emitc_rvv_packed_i4_offset_binary_dot_i8_from_source_rvv_packed_i4_offset_binary_dot_i8(
// CORE: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop i32 seed: out[0] = acc[0].
// CORE: %[[ACCSCALAR:.*]] = load
// CORE: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CORE: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// CORE: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The three i8/mf4 source loads.
// CORE: %[[W:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// CORE: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// CORE: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// Offset-binary -> two's-complement: xor 0x88 the WEIGHT only.
// CORE: %[[WXOR:.*]] = call_opaque "__riscv_vxor_vx_i8mf4"(%[[W]],
// Low nibble: shift into the high nibble then arithmetic-shift back (sign-extend).
// CORE: %[[WLOSH:.*]] = call_opaque "__riscv_vsll_vx_i8mf4"(%[[WXOR]],
// CORE: %[[V0:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WLOSH]],
// High nibble: arithmetic-shift sign-extends it in place.
// CORE: %[[V1:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WXOR]],
// Asymmetric widening product: decoded i8 weight x PLAIN i8 activation halves.
// CORE: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16mf2"(%[[V0]], %[[QLO]], %[[BODYVL]]) : (!emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// CORE: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16mf2"(%[[PROD]], %[[V1]], %[[QHI]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// In-loop i32 running seed + the signed widening reduce (vwredsum).
// CORE: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CORE: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CORE: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// CORE: return

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// WRONGSIG: bounded RVV packed-i4 offset-binary dot source front door failed
// WRONGSIG-SAME: exactly six inputs
