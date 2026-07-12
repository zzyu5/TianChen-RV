// END-TO-END production-export CLOSURE for the C4 (N=3 + ConstantTableLoad codebook)
// front door: the front door's OWN auto-constructed codebook integer-core body now
// flows through the FULL production-export pipeline (--weft-materialize-emission-plans)
// and emits correct EmitC. This is the P1f payoff -- the FIRST N-operand route with a
// non-linear LUT (ConstantTableLoad) source proven end-to-end, and the FIRST route
// whose low-precision resource candidate is ASYMMETRIC-SIGNED (u8 gather-index source,
// signed i16 product / i32 result).
//
// WHY this is a distinct, load-bearing test (vs the front-door fixture
// test/Transforms/RVV/rvv-codebook-gather-dot-source-front-door.mlir): that fixture
// runs front-door --> --weft-rvv-lower-to-emitc DIRECTLY. This test inserts
// --weft-materialize-emission-plans in the MIDDLE -- the production-export chain --
// proving the codebook N=3+LUT route survives emission-plan materialization end-to-end
// (the asymmetric codebook resource candidate + the codebook runtime-ABI list + the
// codebook role-sequence spec are all exercised), not just the direct lower.
//
// THE CHAIN (matches the front-door CORE anchor): a broadcast kvalues table load
//   --> three source loads: w (UNSIGNED u8 packed-i4 gather index) / qlo / qhi (plain i8)
//   --> codebook decode: vand 0x0F / vsrl 0x04 nibble split (of the UNSIGNED weight)
//       --> vrgather through the kvalues table (NO q4_0 xor/sll/sra -- the nibble is an
//           INDEX gathered through a non-linear table, not a linear-decoded value)
//   --> asymmetric widening product: LOW idx gather x qlo (vwmul) then HIGH idx gather x
//       qhi (vwmacc) -- the low->qlo / high->qhi pairing is SSA-pinned below so a swap
//       fails the test (pins the LUT-gather correctness)
//   --> signed widening reduce (vwredsum i16 -> i32m1)
//   --> vse32 store.
//
// CAPABILITY -- the codebook DOES flip (contrast with the C3 offset-binary sibling,
// which is VLEN-invariant). The codebook i8 gather anchor is m1 at VLEN128 (mf2 pruned,
// its VLMAX 8 < the 16-entry table) and mf2 at VLEN256 (the lighter footprint wins). So
// the VLEN128 vs VLEN256 EMITTED C DIFFERS: vle8/vrgather_vv_i8m1 + i16m2 product at
// VLEN128 vs vle8/vrgather_vv_i8mf2 + i16m1 product at VLEN256. The EMITC256-NOT lines
// pin the flip in both directions so an accidental non-flip fails the test.
//
// HOST-ONLY: correctness of the codebook decode + asymmetric product + widening reduce
// is covered by the existing host scalar oracle; this asserts only that the C4 codebook
// front-door output now EXPORTS (exit 0) through the production plan chain with the
// correct 6-arg signature (UNSIGNED w) + codebook intrinsics. NO board / NO perf claim.

// VLEN128 production-export: front door auto-constructs the body, materializes the
// emission plan, lowers to EmitC. The codebook i8 gather anchor is m1.
// RUN: weft-opt %s --weft-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// VLEN256 production-export: the SAME generic source, capability tier rv64gcv_zvl256b.
// The codebook gather FLIPS to the mf2 anchor (i16m1 product).
// RUN: weft-opt %s --weft-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv_zvl256b --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC256

// ===================== VLEN128 EMITTED C4 (N=3 + LUT) CHAIN @ m1 =============
// The 6-arg EmitC signature: w = const UINT8_t* (the UNSIGNED gather index -- the one
// arg that distinguishes C4 from the signed C3 offset-binary route), qlo / qhi =
// const int8_t*, acc = const int32_t*, out = int32_t*, n = size_t.
// EMITC: emitc.func @weft_emitc_rvv_codebook_gather_dot_i8_from_source_rvv_codebook_gather_dot_i8(
// EMITC-SAME: %arg0: !emitc.ptr<!emitc.opaque<"const uint8_t">>,
// EMITC-SAME: %arg1: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC-SAME: %arg2: !emitc.ptr<!emitc.opaque<"const int8_t">>,
// EMITC-SAME: %arg3: !emitc.ptr<!emitc.opaque<"const int32_t">>,
// EMITC-SAME: %arg4: !emitc.ptr<!emitc.opaque<"int32_t">>,
// EMITC-SAME: %arg5: !emitc.opaque<"size_t">)
// The strip vsetvl is e32m1.
// EMITC: call_opaque "__riscv_vsetvl_e32m1"
// The structured kvalues table decl (the ConstantTableLoad source). A mutation of ANY
// entry fails this line -- pinning the LUT contents.
// EMITC: verbatim "static const int8_t weft_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The table broadcast-load into the values vreg (i8m1) the gather indexes.
// EMITC: %[[VALUES:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// The three source loads: w (UNSIGNED u8m1) / qlo (i8m1) / qhi (i8m1).
// EMITC: %[[W:.*]] = call_opaque "__riscv_vle8_v_u8m1"
// EMITC: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// EMITC: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// Codebook nibble split of the UNSIGNED weight: low nibble = vand 0x0F, high = vsrl 0x04.
// EMITC: %[[IDXLO:.*]] = call_opaque "__riscv_vand_vx_u8m1"(%[[W]],
// EMITC: %[[IDXHI:.*]] = call_opaque "__riscv_vsrl_vx_u8m1"(%[[W]],
// The codebook is a NON-LINEAR gather -- NO q4_0 xor/sll/sra linear decode.
// EMITC-NOT: call_opaque "__riscv_vxor_vx_i8m1"
// EMITC-NOT: call_opaque "__riscv_vsra_vx_i8m1"
// The two vrgather codebook lookups: gather VALUES at IDXLO -> V0, at IDXHI -> V1.
// A swap of IDXLO/IDXHI or a wrong table operand fails these captures.
// EMITC: %[[V0:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXLO]],
// EMITC: %[[V1:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXHI]],
// PAIRING PIN: LOW gather (%[[V0]]) x qlo (%[[QLO]]) via vwmul; HIGH gather (%[[V1]]) x
// qhi (%[[QHI]]) via vwmacc. A low<->high or qlo<->qhi swap fails these captures.
// EMITC: call_opaque "__riscv_vwmul_vv_i16m2"(%[[V0]], %[[QLO]], %{{.*}}) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// EMITC: call_opaque "__riscv_vwmacc_vv_i16m2"(%{{.*}}, %[[V1]], %[[QHI]], %{{.*}}) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// The signed widening reduce (vwredsum i16m2 -> i32m1) + store.
// EMITC: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMITC: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// EMITC: return

// ===================== VLEN256 EMITTED C4 CHAIN -- THE FLIP @ mf2 ============
// The SAME 6-arg signature (UNSIGNED w), the SAME kvalues table, but the codebook gather
// FLIPS to the mf2 anchor with an i16m1 product -- byte-different C from VLEN128.
// EMITC256: emitc.func @weft_emitc_rvv_codebook_gather_dot_i8_from_source_rvv_codebook_gather_dot_i8(
// EMITC256-SAME: %arg0: !emitc.ptr<!emitc.opaque<"const uint8_t">>,
// EMITC256-SAME: %arg5: !emitc.opaque<"size_t">)
// EMITC256: verbatim "static const int8_t weft_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The table + source loads flip to mf2 -- NOT the m1 VLEN128 forms.
// EMITC256: %[[VALUES2:.*]] = call_opaque "__riscv_vle8_v_i8mf2"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_i8m1"
// EMITC256: %[[W2:.*]] = call_opaque "__riscv_vle8_v_u8mf2"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_u8m1"
// EMITC256: %[[QLO2:.*]] = call_opaque "__riscv_vle8_v_i8mf2"
// EMITC256: %[[QHI2:.*]] = call_opaque "__riscv_vle8_v_i8mf2"
// Same codebook nibble split (on the unsigned weight), now u8mf2.
// EMITC256: %[[IDXLO2:.*]] = call_opaque "__riscv_vand_vx_u8mf2"(%[[W2]],
// EMITC256: %[[IDXHI2:.*]] = call_opaque "__riscv_vsrl_vx_u8mf2"(%[[W2]],
// The mf2 gathers -- NOT the m1 form. Same LUT-pairing pin.
// EMITC256: %[[V02:.*]] = call_opaque "__riscv_vrgather_vv_i8mf2"(%[[VALUES2]], %[[IDXLO2]],
// EMITC256-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMITC256: %[[V12:.*]] = call_opaque "__riscv_vrgather_vv_i8mf2"(%[[VALUES2]], %[[IDXHI2]],
// PAIRING PIN (VLEN256): LOW x qlo, HIGH x qhi -- product FLIPS to i16m1, NOT i16m2.
// EMITC256: call_opaque "__riscv_vwmul_vv_i16m1"(%[[V02]], %[[QLO2]],
// EMITC256-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMITC256: call_opaque "__riscv_vwmacc_vv_i16m1"(%{{.*}}, %[[V12]], %[[QHI2]],
// EMITC256-NOT: call_opaque "__riscv_vwmacc_vv_i16m2"
// The reduce flips to i16m1 -> i32m1 source -- NOT the i16m2 VLEN128 form.
// EMITC256: call_opaque "__riscv_vwredsum_vs_i16m1_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMITC256: return

module attributes {weft_rvv.source_front_door = "bounded_codebook_gather_dot_source"} {
  func.func @source_codebook_dot(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>, %acc: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}
