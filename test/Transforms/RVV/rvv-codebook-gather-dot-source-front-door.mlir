// Track B G2, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the codebook
// (vrgather) INTEGER-CORE body from a marked GENERIC codebook-core source, instead
// of routing a monolithic op to a per-kernel hand emitter. The auto-constructed
// body is the single-strip generic-op composition
//   codebook_table_broadcast (the 16-entry kvalues lookup table -> values vreg)
//   load x3 (UNSIGNED packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> tcrv_rvv.codebook_gather_x_i8_product (the nibble split + vrgather codebook
//        decode + asymmetric widening product)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> tcrv_rvv.store
// and it lowers to the SAME codebook decode chain the existing iq4_nl block-dot
// emitter pins (vand 0x0F / vsrl 0x04 nibble split -> vrgather_vv_i8 through the
// broadcast table -> vwmul/vwmacc -> vwredsum), proving the generic
// auto-construction mechanism (proven for the q4_0 nibble in G1) REACHES a
// STRUCTURALLY DIFFERENT family: the codebook does NOT decode linearly -- each
// nibble is an INDEX gathered through a non-linear table (NO q4_0 xor-0x88/sll/sra).
//
// HONEST SCOPE -- the codebook integer CORE ONLY (NOT the full codebook KERNEL).
// There is NO nb = n / QK outer block loop, NO per-block fp16 scale read, NO fp32
// fold, and NO once-above-loop table hoisting (the table_broadcast is per-strip
// here); those need NEW generic ODS vocabulary -- full G2, DEFERRED. The monolithic
// tcrv_rvv.iq4_nl_q8_0_block_dot / mxfp4 / nvfp4 ops, their KERNEL front doors, and
// the hand emitters all STAY.
//
// CAPABILITY -- the codebook DOES flip (the q4_0 sibling did NOT). The codebook i8
// gather anchor is SELECTED by the shared schedule authority and THREADED into the
// body types: m1 at VLEN128 (mf2 PRUNED, its VLMAX 8 < the 16-entry table), mf2 at
// VLEN256 (admitted, the lighter footprint wins). The VLEN128 vs VLEN256 emit
// DIFFER (vrgather_vv_i8m1 + i16m2 product vs vrgather_vv_i8mf2 + i16m1 product) --
// the genuine capability flip, demonstrated at the bounded-core granularity. This
// is MECHANISM/parity, NOT a speed beat (deferred to G5).

// The auto-constructed codebook integer-core body @ VLEN128 (m1 anchor).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=BODY128
//
// The VLEN128 body lowered to EmitC: the codebook decode chain at the m1 anchor.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=CORE128
//
// The capability FLIP @ VLEN256 (mf2 anchor): the SAME generic source materializes
// a byte-DIFFERENT codebook core (mf2 gather + i16m1 product).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=BODY256
// RUN: tcrv-opt %s --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=CORE256
//
// FAIL-CLOSED (I7), no guaranteed VLEN tier: the codebook gather needs VLMAX >= 16
// to index the 16-entry table; a zve32x profile prunes EVERY anchor.
// RUN: not tcrv-opt %s --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gc_zve32x 2>&1 | FileCheck %s --check-prefix=NOVLEN
//
// FAIL-CLOSED (I7): a marker-carrying source with the WRONG (4-arg) signature is
// REJECTED, not silently lowered.
// RUN: not tcrv-opt %S/Inputs/codebook-gather-dot-source-wrong-signature.mlir --tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv 2>&1 | FileCheck %s --check-prefix=WRONGSIG

module attributes {tcrv_rvv.source_front_door = "bounded_codebook_gather_dot_source"} {
  func.func @source_codebook_dot(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>, %acc: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED CODEBOOK-CORE BODY @ VLEN128 =========
// The generic codebook-core source becomes a tcrv.exec.kernel with the auto-built
// codebook_table_broadcast / load x3 / codebook_gather_x_i8_product /
// standalone_reduce / store body. NO per-kernel emitter authored this body.
// BODY128: tcrv.exec.kernel @rvv_codebook_gather_dot_i8_from_source
// BODY128: tcrv.exec.variant @rvv_codebook_gather_dot_i8
// The THREADED codebook integer-core anchor (audit-only provenance): the m1 flip form.
// BODY128: tcrv_rvv.codebook_integer_core_anchor = "i8m1-i16m2-i32m1-vlen-flip"
// BODY128: tcrv_rvv.codebook_table_broadcast
// BODY128-SAME: table_symbol = "tcrv_iq4_nl_kvalues"
// BODY128-SAME: : !tcrv_rvv.vector<i8, "m1">
// The UNSIGNED packed-i4 weight + the two plain-i8 q8 activation halves (i8/m1).
// BODY128: tcrv_rvv.load
// BODY128-SAME: -> !tcrv_rvv.vector<ui8, "m1">
// BODY128: tcrv_rvv.load
// BODY128-SAME: -> !tcrv_rvv.vector<i8, "m1">
// BODY128: tcrv_rvv.load
// BODY128-SAME: -> !tcrv_rvv.vector<i8, "m1">
// The auto-constructed codebook gather decode + asymmetric widening product (i16/m2).
// BODY128: tcrv_rvv.codebook_gather_x_i8_product
// BODY128-SAME: kind = "signed_codebook_gather_x_i8_product"
// BODY128-SAME: product_relation = "codebook-gather-i8-x-i8x2-to-i16"
// BODY128-SAME: -> !tcrv_rvv.vector<i16, "m2">
// BODY128: tcrv_rvv.standalone_reduce
// BODY128-SAME: -> !tcrv_rvv.vector<i32, "m1">
// BODY128: tcrv_rvv.store
// BODY128: tcrv.exec.variant @rvv_codebook_gather_dot_i8_scalar_fallback
// BODY128: tcrv.exec.case @rvv_codebook_gather_dot_i8
// BODY128: tcrv.exec.fallback @rvv_codebook_gather_dot_i8_scalar_fallback

// ===================== EMITTED CODEBOOK-CORE CHAIN @ VLEN128 (m1) ============
// The auto-constructed body lowers to the codebook decode chain the existing iq4_nl
// block-dot emitter pins: the structured kvalues decl + broadcast, the UNSIGNED
// weight load, the vand/vsrl nibble split, the vrgather codebook gather, the
// asymmetric vwmul/vwmacc product, and the vwredsum widening reduce -- all at m1.
// CORE128: emitc.func @tcrv_emitc_rvv_codebook_gather_dot_i8_from_source_rvv_codebook_gather_dot_i8(
// CORE128: verbatim "static const int8_t tcrv_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// CORE128: %[[VALUES:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CORE128: %[[W:.*]] = call_opaque "__riscv_vle8_v_u8m1"
// CORE128: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CORE128: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CORE128: %[[IDXLO:.*]] = call_opaque "__riscv_vand_vx_u8m1"(%[[W]],
// CORE128: %[[IDXHI:.*]] = call_opaque "__riscv_vsrl_vx_u8m1"(%[[W]],
// CORE128: %[[V0:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXLO]],
// CORE128: %[[V1:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXHI]],
// CORE128-NOT: call_opaque "__riscv_vxor_vx_i8m1"
// CORE128: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m2"(%[[V0]], %[[QLO]],
// CORE128: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16m2"(%[[PROD]], %[[V1]], %[[QHI]],
// CORE128: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%[[PAIR]],

// ===================== THE CAPABILITY FLIP @ VLEN256 (mf2) ===================
// The SAME generic source, the SAME front door, a DIFFERENT capability fact: the
// auto-constructed body anchors at mf2, the i16 product at m1 -- byte-different C.
// BODY256: tcrv_rvv.codebook_integer_core_anchor = "i8mf2-i16m1-i32m1-vlen-flip"
// BODY256: tcrv_rvv.codebook_table_broadcast
// BODY256-SAME: : !tcrv_rvv.vector<i8, "mf2">
// BODY256: tcrv_rvv.load
// BODY256-SAME: -> !tcrv_rvv.vector<ui8, "mf2">
// BODY256: tcrv_rvv.codebook_gather_x_i8_product
// BODY256-SAME: -> !tcrv_rvv.vector<i16, "m1">
//
// The emitted codebook core FLIPS to the mf2 gather + i16m1 product (the genuine
// VLEN128-vs-VLEN256 capability divergence the board-sealed FP4 brick records).
// CORE256: %[[VALUES2:.*]] = call_opaque "__riscv_vle8_v_i8mf2"
// CORE256: %[[W2:.*]] = call_opaque "__riscv_vle8_v_u8mf2"
// CORE256: call_opaque "__riscv_vand_vx_u8mf2"(%[[W2]],
// CORE256: call_opaque "__riscv_vsrl_vx_u8mf2"(%[[W2]],
// CORE256: call_opaque "__riscv_vrgather_vv_i8mf2"(%[[VALUES2]],
// CORE256: call_opaque "__riscv_vrgather_vv_i8mf2"(%[[VALUES2]],
// CORE256: call_opaque "__riscv_vwmul_vv_i16m1"
// CORE256: call_opaque "__riscv_vwmacc_vv_i16m1"
// CORE256: call_opaque "__riscv_vwredsum_vs_i16m1_i32m1"

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// NOVLEN: prunes every legal codebook i8 gather anchor
// NOVLEN-SAME: VLMAX < 16
// WRONGSIG: bounded RVV codebook-gather dot source front door failed
// WRONGSIG-SAME: exactly six inputs
