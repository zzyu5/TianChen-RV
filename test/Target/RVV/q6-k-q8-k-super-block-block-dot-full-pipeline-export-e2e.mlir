// FULL production-export CLOSURE for the q6_K (ggml Q6_K x Q8_K super-block
// block-dot) front door: the front door's OWN auto-constructed monolithic
// super-block block-dot body now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through weft-translate --weft-export-target-artifact. This is the
// q6_K sibling of the q4_K full-pipeline closure: it takes q6_K from "emits
// through the emission-plans stage" to "fully production-reachable" -- coherence
// recognizes the SHARED super-block monolithic route id and the target artifact
// exports byte-identical to the CORE emit.
//
// WHY this is a distinct, load-bearing test (vs the typed super-block CORE emit
// fixture test/Conversion/RVV/rvv-to-emitc-q6-k-q8-k-typed-super-block-block-dot-
// loop-body.mlir): that fixture runs a hand-authored typed-body kernel through
// --weft-rvv-lower-to-emitc DIRECTLY (the CORE emit). THIS test drives the FULL
// weft-source-artifact-front-door-pipeline --
// which auto-constructs the body from a marked operator-identity source, ALSO
// runs --weft-check-execution-plan-coherence with the built-in target-artifact
// exporter registry -- and then exercises the real target-artifact OBJECT export.
// It proves the monolithic route survives coherence + export, not just plan
// materialization.
//
// THE TWO WALLS the shared super-block family closed (both block-dot-generic, not
// q6_K-specific): coherence recognizes the monolithic route id
// 'rvv-ggml-super-block-block-dot-monolithic-emitc-route-family' as a registered
// target-artifact export route (an honest peer OBJECT exporter), and target-
// artifact-export route resolution reaches the monolithic body via the purpose-
// built candidate validator checking the emission plan's honest monolithic fields
// DIRECTLY (no describe-slice), gated narrowly to the monolithic route id --
// decomposed routes are byte-exact untouched. q6_K rides the SAME super-block
// family wiring as q4_K/iq4_xs (its 4-role ggml vec_dot ABI n/s/vx/vy and its
// super-block-ness are what the family keys off; the 6-bit ql+qh unpack + the
// per-16 int8-scaled aux32 + the SYMMETRIC deferred fold -- NO min, NO dmin --
// are op structure the emitter consumes, not a route-family concern).
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same
// tryConvertConstructedModuleWithRegisteredBackend / convertRVVModuleToEmitC lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the exported artifact's emit is
// byte-identical to the CORE == emission-plans emit (the typed super-block
// SINGLE-accumulator lowering emitTypedSuperBlockScalesTimesSumiLoopBody, itself
// byte-identical to the retired monolith emitQ6_KQ8_KBlockDot). The exported
// function symbol is the kernel+variant handoff name. NO perf claim -- this is
// coverage/wiring maturity (q6_K is NOT in any schedule autotuner; the loop lowers
// at the emitter's default mf2 integer-core anchor, the integer_core_lmul knob
// stays dormant).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact;
// the coherence half of this closure is exercised unconditionally by the PLAN run
// below.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic block-dot body, the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND
// passes --weft-check-execution-plan-coherence (the monolithic route id is now a
// registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-q6-k-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// Target-artifact OBJECT export: the monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q6-k-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {weft_rvv.source_front_door = "ggml_q6_K_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q6_K_q8_K_kernel"} {
  func.func @source_q6_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the monolithic route id + object artifact kind.
// q6_K FIRST FLIP: the front door now auto-constructs the typed SUPER-BLOCK
// SINGLE-accumulator loop body (weft_rvv.typed_super_block_block_dot_loop_body,
// fold_model "scales_times_sumi") from the q6_K aux32 integer core + the reused
// no-min positive fold, NOT an opaque weft_rvv.q6_k_q8_k_block_dot op (retired).
// It still resolves to its OWN monolithic super-block export entry (by fold_model
// + weight_block_stride 210), so the route id + ABI + object export are unchanged.
// PLAN: weft.exec.kernel @ggml_vec_dot_q6_K_q8_K_kernel
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The honest monolithic-body route id (NOT the decomposed generic-typed-body
// route) is the coherence-recognized target-artifact export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q6_K_q8_K_block_dot
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
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot
