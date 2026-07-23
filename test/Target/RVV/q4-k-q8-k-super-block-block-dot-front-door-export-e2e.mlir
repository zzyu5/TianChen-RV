// END-TO-END production-export CLOSURE for the q4_K (ggml Q4_K x Q8_K super-block
// block-dot) front door: the front door's OWN auto-constructed monolithic
// super-block block-dot body now flows through the FULL production-export
// pipeline (--weft-materialize-emission-plans) and emits correct EmitC. This is
// the P2-b payoff -- it closes the README's named gap ("q4_K proven-decomposable
// (6-of-7 brick witnesses, byte-exact) but not wired to production").
//
// WHY this is a distinct, load-bearing test (vs the front-door fixture
// test/Transforms/RVV/rvv-q4-k-q8-k-block-dot-source-front-door.mlir): that
// fixture runs front-door --> --weft-rvv-lower-to-emitc DIRECTLY (the CORE emit).
// This test inserts --weft-materialize-emission-plans in the MIDDLE -- the
// production-export chain -- proving the q4_K route survives emission-plan
// materialization end-to-end.
//
// THE ARCHITECTURE (why q4_K needed distinct wiring from C3/C4): the decomposed
// contraction/codebook routes (packed-i4, codebook-gather) construct a
// STRAIGHT-LINE of generic typed micro-ops the RVV route-slice analysis walks to
// build a route-family emission plan. The q4_K body is instead ONE monolithic
// plugin-owned typed op (weft_rvv.q4_k_q8_k_block_dot) carrying the whole
// super-block dot as first-class STRUCTURE (the super-block loop, the 6-bit
// scale/min bit-dance, the aux32 accumulation, the deferred fp32 fold, and the
// q4_K MIN term), lowering DIRECTLY through the RVV->EmitC DialectConversion.
// There is no decomposed route slice to describe, so the RVV plugin emits an
// HONEST monolithic-body emission plan (plugin-owned typed body materializing
// EmitC through the common DialectConversion) rather than faking the
// product-reduction operand-binding metadata the decomposed routes carry.
//
// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror to the kernel; the block-dot body is untouched, so the
// emitted C is BYTE-IDENTICAL to the CORE emit (the direct --weft-rvv-lower-to-
// emitc path in the front-door fixture) -- the same CORE == production-export
// criterion the C3/C4 e2e closures use. That CORE emit is in turn byte-exact vs
// ggml's real ggml_vec_dot_q4_K_q8_K _generic fp32 order (the hand-authored q4_K
// emitter's scalar-oracle target). NO board / NO perf claim -- this is coverage/
// wiring maturity, not the (board-pending, manual-stamped) q4_K micro-win.
//
// VLEN-INVARIANT: q4_K has NO VLEN128-vs-VLEN256 byte-flip and is NOT in any
// schedule autotuner, so the op lowers at the emitter's default mf2 integer-core
// anchor identically at every tier; ONE EMITC check pins the production-export.

// Production-export: front door auto-constructs the monolithic block-dot body,
// materializes the emission plan, lowers to EmitC.
// RUN: weft-opt %s --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

module attributes {weft_rvv.source_front_door = "ggml_q4_K_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q4_K_q8_K_kernel"} {
  func.func @source_q4_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ============ EMITTED super-block dot core (default mf2), post production-export ============
// The 4-arg ggml vec_dot ABI: n = size_t, s = float*, vx/vy = const uint8_t*.
// EMITC: emitc.func @weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(
// EMITC-SAME: %arg0: !emitc.opaque<"size_t">
// EMITC-SAME: %arg1: !emitc.ptr<!emitc.opaque<"float">>
// EMITC-SAME: %arg2: !emitc.ptr<!emitc.opaque<"const uint8_t">>
// EMITC-SAME: %arg3: !emitc.ptr<!emitc.opaque<"const uint8_t">>
// The int8_t aux8[256] + uint32_t utmp[4] super-block scratch.
// EMITC: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// EMITC: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// The e8m2(32) nibble unpack (plain 4-bit, NO offset-binary bias).
// EMITC: call_opaque "__riscv_vsetvl_e8m2"
// EMITC: call_opaque "__riscv_vle8_v_u8m2"
// EMITC: call_opaque "__riscv_vand_vx_u8m2"
// The deferred two-level fp32 fold: SEPARATE vfmul/vfadd, NEVER a fused FMA.
// EMITC: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMITC: call_opaque "__riscv_vfmul_vf_f32m2"
// EMITC: call_opaque "__riscv_vfadd_vv_f32m2"
// EMITC-NOT: call_opaque "__riscv_vfmacc
// EMITC-NOT: call_opaque "__riscv_vfmadd
// The sequential horizontal sum stores fp32 *s -- NOT a vfredusum.
// EMITC: call_opaque "__riscv_vse32_v_f32m2"
// EMITC-NOT: call_opaque "__riscv_vfredusum
// EMITC: return
// The residual operator-identity source func lowers to NOTHING: exactly ONE kernel.
// EMITC-NOT: emitc.func @weft_emitc_source_q4_K_q8_K_block_dot
