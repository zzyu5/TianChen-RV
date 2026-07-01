// FULL production-export CLOSURE for the q4_0 (ggml Q4_0 x Q8_0 FLAT block-dot)
// front door -- the P2-b chunk-3 payoff that brings a SECOND block-dot op (and
// the FIRST FLAT-family one) through the SAME generalized monolithic wiring q4_K
// proved. The front door's OWN auto-constructed monolithic flat block-dot body
// now flows through the COMPLETE tcrv-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --tcrv-check-execution-plan-coherence) AND
// exports a real RISC-V target artifact through tcrv-translate
// --tcrv-export-target-artifact.
//
// CHUNK 3 = the mechanism generalized off a SHARED block-dot family trait
// (RVVMonolithicBlockDotFamily.h), not the q4_K op type. The recognition, the
// SUPER-BLOCK vs FLAT route-family split, and the per-op ABI table are shared;
// q4_0 is a FLAT op, so it takes the flat route id
// 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family' (NOT q4_K's super-block
// route id) and carries the FULL 8-role strided ggml vec_dot ABI (n, s, bs, vx,
// bx, vy, by, nrc) -- the block-dot op consumes only vx/vy/s/n but the exported C
// signature mirrors ggml's prototype. q4_K stays byte-exact on the super-block
// route (its own e2e is unchanged).
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same
// tryConvertModuleWithRegisteredBackend / convertRVVModuleToEmitC lowering the
// direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q4_0's front-door-constructed op is
// left attr-less (no shape knob); the emitter lowers it at its wide-LMUL m1
// default anchor -- there is NO VLEN128-vs-VLEN256 byte-flip for q4_0.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic flat block-dot body,
// the tcrv-source-artifact-front-door-pipeline materializes the emission plan AND
// passes --tcrv-check-execution-plan-coherence (the flat monolithic route id is
// now a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {tcrv_rvv.source_front_door = "ggml_q4_0_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel"} {
  func.func @source_q4_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel
// PLAN: tcrv_rvv.q4_0_q8_0_block_dot
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline).
// PLAN-SAME: rvv_ggml_flat_block_dot_kind
// The honest FLAT monolithic-body route id (NOT q4_K's super-block route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-flat-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q4_0_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata,
// and never claims the super-block route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot
