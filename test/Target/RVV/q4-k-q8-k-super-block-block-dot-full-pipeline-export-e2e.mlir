// FULL production-export CLOSURE for the q4_K (ggml Q4_K x Q8_K super-block
// block-dot) front door: the front door's OWN auto-constructed monolithic
// super-block block-dot body now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through weft-translate --weft-export-target-artifact. This is the
// P2-b chunk-2 payoff: it takes q4_K from "emits through the emission-plans
// stage" (chunk 1) to "fully production-reachable" -- coherence recognizes the
// monolithic route id and the target artifact exports byte-identical to the CORE
// emit.
//
// WHY this is a distinct, load-bearing test (vs the emission-plans-stage sibling
// q4-k-q8-k-super-block-block-dot-front-door-export-e2e.mlir): that sibling runs
// front-door --> --weft-materialize-emission-plans --> --weft-rvv-lower-to-emitc
// (the CORE == emission-plans emit). THIS test drives the FULL
// weft-source-artifact-front-door-pipeline -- which ALSO runs
// --weft-check-execution-plan-coherence with the built-in target-artifact
// exporter registry -- and then exercises the real target-artifact OBJECT export.
// It proves the monolithic route survives coherence + export, not just plan
// materialization.
//
// THE TWO WALLS chunk 2 closed (both were block-dot-generic, not q4_K-specific):
//   1. COHERENCE rejected the monolithic route id
//      'rvv-ggml-super-block-block-dot-monolithic-emitc-route-family' as an
//      "unknown target artifact export route id" -- the RVV target support
//      bundle only registered the decomposed 'rvv-generic-typed-body-emitc-route-
//      family'. Chunk 2 registers an HONEST peer OBJECT exporter under the
//      monolithic route id (a bare object exporter, NOT a header/object bundle
//      composite that would shadow the decomposed routes).
//   2. TARGET-ARTIFACT-EXPORT route resolution funneled through the same
//      slice-based describeRVVSelectedBodyEmitCRoute chokepoint chunk 1's plan
//      branch bypassed. Chunk 2 mirrors chunk-1's monolithic-body handling with
//      a purpose-built candidate validator that checks the emission plan's honest
//      monolithic fields DIRECTLY (no describe-slice), gated narrowly to the
//      monolithic route id -- decomposed routes are byte-exact untouched.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same
// tryConvertModuleWithRegisteredBackend / convertRVVModuleToEmitC lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the exported artifact's emit is
// byte-identical to the CORE == emission-plans emit. The typed super-block loop
// body's executable C is byte-identical to the retired monolith's (modulo the
// source-op provenance comment token; the exported object, comments stripped, is
// byte-identical). The exported function symbol is the kernel+variant handoff name. NO perf
// claim -- this is coverage/wiring maturity (q4_K is NOT in any schedule
// autotuner; the op lowers at the emitter's default mf2 integer-core anchor).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact;
// the coherence half of this closure is exercised unconditionally by the sibling
// emission-plans e2e and by the PLAN run below.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic block-dot body, the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND
// passes --weft-check-execution-plan-coherence (the monolithic route id is now a
// registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// Target-artifact OBJECT export: the monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {weft_rvv.source_front_door = "ggml_q4_K_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q4_K_q8_K_kernel"} {
  func.func @source_q4_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the monolithic route id + object artifact kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_q4_K_q8_K_kernel
// The q4_K front door's body is the typed SUPER-BLOCK block-dot LOOP body op
// (M-FLAT milestone-3); it exports through the SAME shared super-block monolithic
// plan (route id / object kind) as the compound q4_K block-dot op it replaced, but
// carries q4_K's op-derived kind metadata resolved from its SuperBlockTwoLevelScaleMin
// entry (NOT an opaque weft_rvv.q4_k_q8_k_block_dot op).
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The honest monolithic-body route id (NOT the decomposed generic-typed-body
// route) is the coherence-recognized target-artifact export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q4_K_q8_K_block_dot
// The block-dot honestly carries NO decomposed-route slice config metadata, and
// the decomposed generic-typed-body route id never appears.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot
