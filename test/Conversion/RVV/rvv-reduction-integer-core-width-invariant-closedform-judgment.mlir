// MIG-5 / §3.4 JUDGMENT EXPERIMENT (A-line, anti-fake-refactor guard) -- the
// family/width integer-core LMUL selector is now an EXPLICIT NAMED CLOSED FORM
// f(VLEN, sew, blockLen) in the gearbox header
// (getRVVEffectiveWidthInvariantLMUL @ RVVGearboxSchedule.h), NOT the old
// enumerateBlockDotShapeCandidates + selectGenericSchedule argmin fallback. The
// reduction front door (RVVReductionSourceFrontDoor.cpp: selectIntegerCoreLMUL)
// CALLS the closed form; this lit proves the closed form is LOAD-BEARING (承重),
// not a decorative fallback.
//
// PINNED 改判 RECORD (single-variable capability flip -- only the VLEN capability
// (via -march) changes; the SAME generic vector.multi_reduction source):
//   * VLEN128 (rv64gcv):         theta_width = m2  (e8m2 body)  reason=effective_width_invariant
//   * VLEN256 (rv64gcv_zvl256b): theta_width = m1  (e8m1 body)  reason=effective_width_invariant
// m2@VLEN128 and m1@VLEN256 are the SAME 256-bit effective register group
// (VLMAX*sew = 32*8 = 256 bits): the closed form LMUL = blockLen*sew / VLEN keeps
// the register-group width INVARIANT, so a wider VLEN pins the same group at a
// NARROWER LMUL. The output family/width MUST FLIP with the capability -- if a
// stale argmin (or a hardcoded literal) were still driving it, the flip could not
// track the width-invariant equation. The reason is the invariant itself, never a
// cost-model `static_order` ([SEL-1-T5]).
//
// Byte-exact discipline: at the deployed board VLEN the closed form == the old
// argmin pick, so the emitted intrinsics are UNCHANGED (this lit and the committed
// rvv-widening-dot-reduce-source-front-door.mlir assert the SAME e8m2/e8m1 bytes).
// Only FLIPPING the VLEN reveals the 改判.

// (A) BODY-level flip: the setvl integer-core anchor the closed form threaded in.
// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | FileCheck %s --check-prefix=BODY128
// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline | FileCheck %s --check-prefix=BODY256
//
// (B) EMITTED-INTRINSIC flip: the width-invariant anchor carries to the emitted C.
// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC128
// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC256
//
// (C) NO-capability fail-closed: source construction still produces exact P, but
// selected owner construction cannot invent a final plan without c_o.
// RUN: not weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gc --weft-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=NOCAP

module attributes {weft_rvv.source_front_door = "bounded_widening_dot_reduce_source"} {
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

// ===================== (A) BODY anchor flip ================================
// VLEN128 => the width-invariant LMUL is m2 (VLMAX e8m2 = 128*2/8 = 32 == blockLen).
// BODY128: weft_rvv.setvl
// BODY128-SAME: lmul = "m2"
// BODY128-SAME: sew = 8
// BODY128: weft_rvv.widening_product
// BODY128-SAME: -> !weft_rvv.vector<i16, "m4">
//
// VLEN256 => the SAME source FLIPS to m1 (VLMAX e8m1 = 256/8 = 32 == blockLen): the
// same 256-bit group at a narrower LMUL. NOT m2.
// BODY256: weft_rvv.setvl
// BODY256-SAME: lmul = "m1"
// BODY256-SAME: sew = 8
// BODY256-NOT: lmul = "m2"
// BODY256: weft_rvv.widening_product
// BODY256-SAME: -> !weft_rvv.vector<i16, "m2">

// ===================== (B) EMITTED-INTRINSIC flip =========================
// VLEN128: e8m2 strip, i16m4 product, vwredsum_i16m4_i32m1 -- NOT the e8m1 forms.
// EMITC128: call_opaque "__riscv_vsetvl_e8m2"
// EMITC128-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMITC128: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
//
// VLEN256: the capability FACT flips the bytes to e8m1 / i16m2 / vwredsum_i16m2.
// EMITC256: call_opaque "__riscv_vsetvl_e8m1"
// EMITC256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// EMITC256: call_opaque "__riscv_vwmul_vv_i16m2"
// EMITC256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"

// ===================== (C) NO-capability zero-regression ==================
// rv64gc (no V, no guaranteed VLEN>=128): no final typed plan is invented.
// NOCAP: reduction formula requires minimum_vlen and vreg_count in c_o
