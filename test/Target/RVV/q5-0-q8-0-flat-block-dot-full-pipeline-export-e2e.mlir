// FULL production-export CLOSURE for the q5_0 (ggml Q5_0 x Q8_0 FLAT block-dot)
// front door -- the P2-d batch that brings the three common legacy FLAT formats
// (q4_1/q5_0/q5_1) through the SAME generalized monolithic wiring q4_K/q4_0 proved.
// The front door's OWN auto-constructed monolithic flat block-dot body flows through
// the COMPLETE tcrv-source-artifact-front-door-pipeline (materialize-emission-plans
// PLUS --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through tcrv-translate --tcrv-export-target-artifact.
//
// CHUNK = the mechanism generalized off a SHARED block-dot family trait
// (RVVMonolithicBlockDotFamily.h), not any one op type. q5_0 is a FLAT op, so it
// takes the flat route id 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family'
// and carries the 4-role ggml vec_dot ABI (n, s, vx, vy) -- same ABI arity as iq4_nl,
// a DISTINCT op kind (Family-A 5-bit: the per-element qh high-bit plane is first-class
// op structure). q4_K stays byte-exact on the super-block route; q4_0 shares the flat
// route with its own 8-role strided ABI (looked up per-op by kind).
//
// SCHEDULE STAMP: like iq4_nl (and unlike q4_0/q4_K's default-anchor lowering), the
// front-door-constructed attr-less op is shaped by the EXISTING
// --tcrv-rvv-materialize-q5-0-schedule gearbox (rv64gcv) before lowering -- the same
// gearbox its CORE emit fixture uses. This is the honest per-op pipeline, not new
// wiring.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC, so the
// production-export emit is byte-identical to the CORE == emission-plans emit
// (asserted below by diff, both under the same schedule stamp). NO board / NO perf
// claim -- coverage/wiring maturity only.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic flat block-dot body, the
// schedule gearbox stamps the integer-core shape, the
// tcrv-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --tcrv-check-execution-plan-coherence.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {tcrv_rvv.source_front_door = "ggml_q5_0_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel"} {
  func.func @source_q5_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel
// The q5_0 front door's body is the typed flat block-dot LOOP body op (M-FLAT);
// it exports through the SAME shared Flat monolithic plan (route id / object kind)
// as the compound q5_0 block-dot op it replaced, but carries q5_0's OWN 4-role ABI
// + scales_times_sumi op-derived kind metadata (NOT q8_0/q4_0's 8-role default).
// PLAN: tcrv_rvv.typed_flat_block_dot_loop_body
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
// PLAN-SAME: target = @rvv_q5_0_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route or a decomposed route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot
