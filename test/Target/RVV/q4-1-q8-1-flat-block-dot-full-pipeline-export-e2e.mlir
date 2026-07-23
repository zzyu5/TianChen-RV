// FULL production-export CLOSURE for the q4_1 (ggml Q4_1 x Q8_1 FLAT block-dot)
// front door -- the P2-d batch that brings the three common legacy FLAT formats
// (q4_1/q5_0/q5_1) through the SAME generalized monolithic wiring q4_K/q4_0 proved.
// The front door's OWN auto-constructed monolithic flat block-dot body flows through
// the COMPLETE weft-source-artifact-front-door-pipeline (materialize-emission-plans
// PLUS --weft-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through weft-translate --weft-export-target-artifact.
//
// CHUNK = the mechanism generalized off a SHARED block-dot family trait
// (RVVMonolithicBlockDotFamily.h), not any one op type. q4_1 is a FLAT op, so it
// takes the flat route id 'rvv-generic-typed-body-emitc-route-family'
// and carries the 4-role ggml vec_dot ABI (n, s, vx, vy) -- same ABI arity as iq4_nl,
// a DISTINCT op kind (Family-B scale+MIN: the per-block MIN plane + the unsigned
// nibble core are first-class op structure). q4_K stays byte-exact on the super-block
// route; q4_0 shares the flat route with its own 8-role strided ABI (looked up per-op
// by kind).
//
// SCHEDULE STAMP: like iq4_nl (and unlike q4_0/q4_K's default-anchor lowering), the
// front-door-constructed attr-less op is shaped by the EXISTING
// --weft-rvv-materialize-q4-1-schedule gearbox (rv64gcv) before lowering -- the same
// gearbox its CORE emit fixture uses. This is the honest per-op pipeline, not new
// wiring.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC, so the
// production-export emit is byte-identical to the CORE == emission-plans emit
// (asserted below by diff, both under the same schedule stamp). NO board / NO perf
// claim -- coverage/wiring maturity only.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic flat block-dot body, the
// schedule gearbox stamps the integer-core shape, the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --weft-check-execution-plan-coherence.
// RUN: weft-opt %s --weft-rvv-materialize-q4-1-q8-1-block-dot-source-front-door=march=rv64gcv --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-q4-1-q8-1-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-q4-1-q8-1-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q4-1-q8-1-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {weft_rvv.source_front_door = "ggml_q4_1_q8_1_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel"} {
  func.func @source_q4_1_q8_1_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel
// The q4_1 front door's body is the typed flat block-dot LOOP body op (M-FLAT);
// it exports through the SAME shared Flat monolithic plan (route id / object kind)
// as the compound q4_1 block-dot op it replaced, but carries q4_1's OWN 4-role ABI
// + scale+MIN op-derived kind metadata (NOT q8_0/q4_0's 8-role default).
// PLAN: weft_rvv.typed_flat_block_dot_loop_body
// PLAN: weft.exec.diagnostic {{.*}}artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline).
// The honest FLAT monolithic-body route id (NOT q4_K's super-block route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-generic-typed-body-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q4_1_q8_1_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route or a decomposed route.

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_rvv_q4_1_q8_1_block_dot
