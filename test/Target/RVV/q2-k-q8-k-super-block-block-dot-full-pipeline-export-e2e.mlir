// FULL production-export CLOSURE for the q2_K (ggml Q2_K x Q8_K super-block
// block-dot) front door: the front door's OWN auto-constructed super-block
// block-dot body now flows through the COMPLETE
// tcrv-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through tcrv-translate --tcrv-export-target-artifact. This is the
// q2_K sibling of the q4_K/q6_K full-pipeline closures: coherence recognizes the
// SHARED super-block monolithic route id and the target artifact exports
// byte-identical to the CORE emit.
//
// q2_K FIRST FLIP: q2_K is a super-block affine quant (it HAS a min term, like
// q4_K/q5_K) but its whole fold is a SINGLE per-super-block SCALAR
// `sumf += dall*isum - dmin*summs`, so the front door now auto-constructs the
// typed SUPER-BLOCK SCALAR-accumulator loop body
// (tcrv_rvv.typed_super_block_block_dot_loop_body, fold_model "scalar_scale_min")
// from the q2_K integer core (the 2-bit unpack + PLAIN uint4-nibble scale/min +
// per-sub-block scalar i32 dot producing the two scalar states isum + summs) + a
// SINGLE `sumf` scalar yield -- NOT an opaque tcrv_rvv.q2_k_q8_k_block_dot op
// (retired the same action as this flip; the scalar fold is emitter-inlined). It
// still resolves to its OWN monolithic super-block export entry (by fold_model +
// weight_block_stride 84), so the route id + ABI + object export are unchanged.
//
// WHY this is a distinct, load-bearing test (vs the typed super-block CORE emit
// fixture test/Conversion/RVV/rvv-to-emitc-q2-k-q8-k-typed-super-block-block-dot-
// loop-body.mlir): that fixture runs a hand-authored typed-body kernel through
// --tcrv-rvv-lower-to-emitc DIRECTLY (the CORE emit + the byte-exact landmarks).
// THIS test drives the FULL tcrv-source-artifact-front-door-pipeline -- which
// auto-constructs the body from a marked operator-identity source, ALSO runs
// --tcrv-check-execution-plan-coherence with the built-in target-artifact
// exporter registry -- and then exercises the real target-artifact OBJECT export.
// It proves the super-block route survives coherence + export, not just plan
// materialization.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --tcrv-rvv-lower-to-emitc path uses), so the exported artifact's emit
// is byte-identical to the CORE == emission-plans emit (the typed super-block
// SCALAR-accumulator lowering emitTypedSuperBlockScalarScaleMinLoopBody, itself
// byte-identical to the retired monolith emitQ2_KQ8_KBlockDot). The exported
// function symbol is the kernel+variant handoff name. NO perf claim -- this is
// coverage/wiring maturity (q2_K is NOT in any schedule autotuner; the loop lowers
// at the emitter's fixed e8m1/i16m2/i32m1 integer-core anchor, q2_K carries NO
// shape knob).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact;
// the coherence half of this closure is exercised unconditionally by the PLAN run
// below.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the super-block block-dot body, the
// tcrv-source-artifact-front-door-pipeline materializes the emission plan AND
// passes --tcrv-check-execution-plan-coherence (the super-block monolithic route
// id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// Target-artifact OBJECT export: the super-block emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {tcrv_rvv.source_front_door = "ggml_q2_K_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q2_K_q8_K_kernel"} {
  func.func @source_q2_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the super-block monolithic route id + object
// artifact kind. q2_K FIRST FLIP: the front door now auto-constructs the typed
// SUPER-BLOCK SCALAR-accumulator loop body (tcrv_rvv.typed_super_block_block_dot_
// loop_body, fold_model "scalar_scale_min") from the q2_K integer core + the
// emitter-inlined scalar fold, NOT an opaque tcrv_rvv.q2_k_q8_k_block_dot op
// (retired). It still resolves to its OWN monolithic super-block export entry (by
// fold_model + weight_block_stride 84), so the route id + ABI + object export are
// unchanged.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q2_K_q8_K_kernel
// PLAN: tcrv_rvv.typed_super_block_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The honest super-block monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q2_K_q8_K_block_dot
// The block-dot honestly carries NO decomposed-route slice config metadata, and
// the decomposed generic-typed-body route id never appears, and never claims the
// flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot
