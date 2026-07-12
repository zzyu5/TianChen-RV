// FULL production-export CLOSURE for the q1_0 (ggml Q1_0 x Q8_0 FLAT block-dot)
// front door -- the LAST flat block-dot family member flipped dispatch-wired ->
// CONSTRUCTED (C_construct 26->27). The front door's OWN auto-constructed TYPED
// flat block-dot LOOP body (weft_rvv.typed_flat_block_dot_loop_body, fold_model
// "flat_binary_two_level") now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// THE FLIP (why this exemplar matters): q1_0 is the genuine STRUCTURAL-SPECIAL
// case of the flat family -- the BINARY {-1,+1}-sign class whose per-super-block
// contribution is a FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL
// fp32 fold (`d0 * Σ_k(d1_k * sumi_block_k)`) that no existing single-core flat
// brick chain (q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) expresses. So UNLIKE those siblings
// its front door constructs the typed flat loop body out of ONE net-new decomposed
// brick -- the q1_0 BINARY-sign INTEGER CORE (weft_rvv.q1_0_q8_0_binary_sign_core:
// the four q8_0 sub-blocks' vlm_v_b{ratio} packed-bit sign mask loaded DIRECTLY as
// the i8 sign mask + i8-domain vneg/vmerge -> ONE vwredsum i8->i16m1, plus the
// emitter-inlined two-level fold) -- the super-block scalar-core precedent
// (tq1_0/iq1_s) applied to the FLAT loop op. q1_0's activation is a FLAT
// block_q8_0 stream, so it still EXPORTS through the EXISTING flat route family
// ('rvv-ggml-flat-block-dot-monolithic-emitc-route-family') and carries the SAME
// 4-role ggml vec_dot ABI (n, s, vx, vy) as iq4_nl -- resolved by its OWN
// typedFlatLoopSelector (Q10BinarySign, keyed off the UNIQUE fold_model), NOT the
// 8-role q8_0 default. The emission is BYTE-IDENTICAL to the retired-in-production
// monolith emitQ1_0Q8_0BlockDot (both call the shared emitQ1_0BlockDotBodyShared),
// modulo only the source-op provenance token. All sibling block-dot ops stay
// byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q1_0's front-door-constructed op is left
// attr-less (no shape knob); q1_0 IS in the unified schedule autotuner (the anchor
// MOVES with VLEN: m2@VLEN128 -> m1@VLEN256), but the front door leaves the
// integer_core_lmul knob unstamped, so the op lowers at its DEFAULT m2 integer-core
// anchor (the VLEN-universal-safe floor: e8m2 VLMAX 32 spans the 32-element sub-block,
// byte-exact at VLEN128) -- stamping the m1 VLEN256 anchor is a separate
// schedule-descriptor concern, not this coverage closure.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the TYPED flat block-dot loop body, the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --weft-check-execution-plan-coherence (the flat monolithic route id is a registered
// target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module attributes {weft_rvv.source_front_door = "ggml_q1_0_q8_0_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q1_0_q8_0_kernel"} {
  func.func @source_q1_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_q1_0_q8_0_kernel
// PLAN: weft_rvv.typed_flat_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline), with the q1_0 kind.
// PLAN-SAME: rvv_ggml_flat_block_dot_kind
// PLAN-SAME: ggml_q1_0_q8_0_block_dot
// The honest FLAT monolithic-body route id (NOT q4_K's super-block route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-flat-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q1_0_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route or a decomposed route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC binary-sign core @ VLEN128 (default m2) ======
// The CORE EmitC is the q1_0 BINARY {-1,+1}-sign decode: each weight bit is a SIGN
// (set -> +q8, clear -> -q8) and the q8 value is the magnitude. The 4 packed bit-bytes
// load DIRECTLY into the i8 sign mask via vlm_v_b{ratio} (the bits ARE the mask), the
// 32 q8 quants are negated/merged IN THE i8 DOMAIN, and ONE vwredsum widens i8 -> i16m1
// per sub-block -- pinned so a regression into a codebook gather, an offset-binary
// nibble decode, or the old kmask/8-lane form is caught.
// CORE: emitc.func @weft_emitc_ggml_vec_dot_q1_0_q8_0_kernel_rvv_q1_0_q8_0_block_dot(
// The function-scoped fp32 accumulator + the SUPER-block count nb = n / 128.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The kmask table is GONE (the packed bits ARE the mask via vlm_v_b{ratio}); the old
// vand/vmsne/vwcvt kmask decode must be ABSENT.
// CORE-NOT: weft_q1_0_kmask
// CORE-NOT: call_opaque "__riscv_vmv_v_x_u8m
// CORE-NOT: call_opaque "__riscv_vand_vv_u8m
// CORE-NOT: call_opaque "__riscv_vmsne_vx_u8m
// CORE-NOT: call_opaque "__riscv_vwcvt_x_x_v_i16
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-super-block weight address (vx + ib*18) + the d0 fp16 read.
// CORE: mul %{{.*}}, %{{.*}}
// CORE: call_opaque "(float)*(const _Float16 *)"
// The per-super-block float accumulator (sumi) RESET to 0.0f.
// CORE: literal "0.0f"
// The q8 sub-block address (vy + (ib*4 + k)*34) + the d1_k fp16 read.
// CORE: call_opaque "(float)*(const _Float16 *)"
// The 32-lane sub-block setvl at the DEFAULT m2 anchor (e8m2 VLMAX 32 spans the block).
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// The BINARY sign decode: vlm_v_b4 the 4 packed bit-bytes DIRECTLY into the i8 mask.
// CORE: call_opaque "__riscv_vlm_v_b4"
// The 32 q8 quants, then negate + merge IN THE i8 DOMAIN (ggml's _vl128 exact ops).
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vneg_v_i8m2"
// CORE: call_opaque "__riscv_vmerge_vvm_i8m2"
// The q4_0 offset-binary nibble decode must be ABSENT (this is a binary sign plane).
// CORE-NOT: call_opaque "__riscv_vxor_vx_i8
// ONE vwredsum per sub-block, widening i8m2 -> i16m1 (seeded by zero) + extract.
// CORE: call_opaque "__riscv_vmv_v_x_i16m1"
// CORE: call_opaque "__riscv_vwredsum_vs_i8m2_i16m1"
// CORE: call_opaque "__riscv_vmv_x_s_i16m1_i16"
// The per-sub-block fold: sumi = sumi + d1 * (float)sumi_block (ONE emitc.expression).
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The super-block fold: sumf = sumf + d0 * sumi.
// CORE: expression : !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store, then return (bounds the checks to the RVV variant body).
// CORE: subscript
// CORE: assign
// CORE: return

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name the
// CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q1_0_q8_0_kernel_rvv_q1_0_q8_0_block_dot
