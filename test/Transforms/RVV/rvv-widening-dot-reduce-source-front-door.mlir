// Track B auto-lowering, FIRST BLOCK -- the BAR-A capability flip. The COMPILER
// auto-CONSTRUCTS the tcrv_rvv RVV-dialect widening int8 dot-reduce body from a
// GENERIC vector-dialect source (vector.transfer_read x2 + arith.extsi x2 +
// arith.muli + vector.multi_reduction <add> + scalar memref.store), instead of a
// per-kernel hand emitter. The integer-core LMUL anchor is the RETURN VALUE of
// the shared block-dot schedule authority (enumerateBlockDotShapeCandidates +
// selectGenericSchedule) fed deriveMinimumVLEN(march) -- NOT a hand switch. So
// the SAME generic source emits a BYTE-DIFFERENT RVV body by capability:
//   * VLEN128 (rv64gcv):        e8m2 -> i16m4 -> vwredsum_i16m4_i32m1
//   * VLEN256 (rv64gcv_zvl256b): e8m1 -> i16m2 -> vwredsum_i16m2_i32m1
// This is the e8m2/e8m1 anchor flip the q8_0 brick #1 shows, but from a GENERIC
// vector.multi_reduction with no per-kernel emitter -- the backend auto-generates
// (not hand-writes) a capability-tuned lowering and produces EmitC.
//
// Framing: vector->RVV lowering exists upstream; the NOVELTY is the capability-
// fact-driven LMUL SELECTION fused into this lowering, reusing the gearbox
// capability facts -- not "we built a Triton backend".
//
// Scope honesty: this is ONE bounded contraction (K=32 signed int8 dot-reduce,
// robust strip form, multi_block_factor pinned to 1). It is the first auto-
// lowered block, not a general dot-reduce auto-tuner.

// The auto-constructed body (the e8m2 VLEN128 anchor).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=BODY
//
// The BAR-A byte-anchor FLIP: the SAME generic source emits e8m2 at VLEN128 and
// e8m1 at VLEN256 -- a BYTE-DIFFERENT emitted kernel driven by the deriveMinimumVLEN
// capability fact through the gearbox authority. Asserted on the INTRINSICS (not a
// metadata mirror), mirroring the q8_0 divergence lit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

module attributes {tcrv_rvv.source_front_door = "bounded_widening_dot_reduce_source"} {
  func.func @source_widening_dot_reduce(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>, %acc: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %seed = memref.load %acc[%c0] : memref<?xi32>
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
    %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
    %p = arith.muli %ae, %be : vector<32xi32>
    %r = vector.multi_reduction <add>, %p, %seed [0] : vector<32xi32> to i32
    memref.store %r, %out[%c0] : memref<?xi32>
    return
  }
}

// ===================== AUTO-CONSTRUCTED tcrv_rvv BODY =======================
// The generic vector source becomes a tcrv.exec.kernel with the auto-built
// load/widening_product/standalone_reduce/store dot-reduce body. NO per-kernel
// emitter authored this -- the front-door matcher constructed it from the VLEN128
// gearbox-selected m2 byte anchor.
// BODY: tcrv.exec.kernel @rvv_widening_dot_reduce_i8_from_vector_source
// BODY: tcrv.exec.variant @rvv_widening_dot_reduce_i8
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m2"
// BODY-SAME: sew = 8
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "m2">
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "m2">
// BODY: tcrv_rvv.widening_product
// BODY-SAME: signed-i8m2xi8m2-to-i16m4
// BODY-SAME: -> !tcrv_rvv.vector<i16, "m4">
// BODY: tcrv_rvv.standalone_reduce
// BODY-SAME: kind = "signed_widening_reduce_add"
// BODY-SAME: -> !tcrv_rvv.vector<i32, "m1">
// BODY: tcrv_rvv.store
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_widening_dot_reduce_i8_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_widening_dot_reduce_i8
// BODY: tcrv.exec.fallback @rvv_widening_dot_reduce_i8_scalar_fallback

// ===================== VLEN128 (rv64gcv) -- the e8m2 anchor =================
// The gearbox selects the m2 byte anchor at the VLEN-128 tier -- vsetvl_e8m2, i8m2
// loads, the i16m4 widening product, the vwredsum_i16m4_i32m1 reduce.
// VLEN128: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_i8_from_vector_source_rvv_widening_dot_reduce_i8(
// VLEN128: call_opaque "__riscv_vsetvl_e8m2"
// VLEN128-NOT: call_opaque "__riscv_vsetvl_e8m1"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) -- the FLIP to e8m1 ========
// The REAL VLEN fact FLIPS the anchor to m1: a BYTE-DIFFERENT kernel. vsetvl_e8m1
// (NOT e8m2), i8m1 loads, the i16m2 product (NOT i16m4), the vwredsum_i16m2_i32m1
// reduce (NOT i16m4). The capability FACT changes the emitted bytes -- bar A.
// VLEN256: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_i8_from_vector_source_rvv_widening_dot_reduce_i8(
// VLEN256: call_opaque "__riscv_vsetvl_e8m1"
// VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN256: return
