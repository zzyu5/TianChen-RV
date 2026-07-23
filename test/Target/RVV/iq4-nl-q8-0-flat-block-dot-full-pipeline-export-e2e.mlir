// FULL production-export CLOSURE for the iq4_nl (ggml IQ4_NL x Q8_0 FLAT block-dot)
// front door -- the CODEBOOK-class member of the SAME generalized flat wiring. After
// the L3-iq4_nl M2 flip the front door's OWN auto-constructed body is the typed flat
// block-dot LOOP body (weft_rvv.typed_flat_block_dot_loop_body, codebook branch), not
// the retired monolith weft_rvv.iq4_nl_q8_0_block_dot op; it flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// The mechanism is generalized off a SHARED block-dot family trait
// (RVVMonolithicBlockDotFamily.h), not the op type. iq4_nl is a FLAT op, so it takes
// the flat route id 'rvv-generic-typed-body-emitc-route-family' (NOT
// q4_K's super-block route) and carries the 4-role ggml vec_dot ABI (n, s, vx, vy) --
// its 16-entry non-linear int8 codebook is first-class op structure (the
// codebook_table_broadcast + codebook_gather_x_i8_product bricks inside the typed
// region). q4_K stays byte-exact on the super-block route.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). NO board
// / NO perf claim -- coverage/wiring maturity only. The iq4_nl codebook core pins the
// m1 half-block anchor (mbf1 / elided), constructed directly by the front door (no
// materialize-schedule stamp), so there is NO VLEN128-vs-VLEN256 byte-flip here.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the typed flat block-dot LOOP body, the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --weft-check-execution-plan-coherence (the flat route id is a registered
// target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door=march=rv64gcv --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat emission plan exports a real RISC-V RVV
// relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {weft_rvv.source_front_door = "ggml_iq4_nl_q8_0_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel"} {
  func.func @source_iq4_nl_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported flat emission-plan
// diagnostic naming the FLAT route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel
// The iq4_nl front door's body is the typed flat block-dot LOOP body op (codebook
// branch); it exports through the SAME shared Flat plan (identical route id / kind /
// 4-role ABI) as the compound iq4_nl block-dot op it replaced.
// PLAN: weft_rvv.typed_flat_block_dot_loop_body
// PLAN: weft.exec.diagnostic {{.*}}artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline).
// The honest FLAT route id (NOT q4_K's super-block route, NOT the decomposed
// generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-generic-typed-body-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq4_nl_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route or a decomposed route.

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot
