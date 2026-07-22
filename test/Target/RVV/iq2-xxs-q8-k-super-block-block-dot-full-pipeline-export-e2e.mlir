// FULL production-export CLOSURE for the iq2_xxs (ggml IQ2_XXS x Q8_K super-block
// GRID-CODEBOOK block-dot) front door -- the coverage payoff that closes the FINAL
// iq* super-block-codebook bucket, the last literal block-dot in the zoo: the
// GRID-codebook class (the packed uint64 iq2xxs_grid[256] + the ksigns_iq2xs[128]
// sign plane, vs iq4_xs's 16-entry per-nibble codebook). The front door's OWN
// auto-constructed monolithic super-block grid-codebook body now flows through the
// COMPLETE weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-grid-codebook is
// MECHANICAL, NOT bespoke. iq2_xxs takes the EXISTING super-block monolithic route
// family (shared with q4_K, the 'rvv-ggml-super-block-block-dot-monolithic-emitc-
// route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The grid + ksigns are OP attrs consumed by the emitter,
// NOT a route-family concern: the emission plan (buildMonolithicBlockDotEmission
// Plan) and the target-export candidate validator key ONLY off op name -> route
// family + the kind/scale_model attrs + the ordered ABI roles; neither reads the grid
// or ksigns. So the shared SuperBlock family + iq2_xxs's grid/ksigns-attr stamping
// COMPOSE cleanly: COVERAGE = one table row (RVVMonolithicBlockDotFamily.h,
// SuperBlock + the 4-role ggml vec_dot ABI n/s/vx/vy) + one front door
// (RVVIQ2XXSBlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/iq4_xs/tq2_0/
// q4_0/q8_0/iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq2_xxs's front-door-constructed op is
// left attr-less (no integer_core_lmul shape knob). UNLIKE q4_K/iq4_xs, iq2_xxs IS in
// a schedule autotuner (m2 at VLEN128, m1 at VLEN256), so there IS a VLEN128-vs-
// VLEN256 flip -- but the attr-less op lowers at the iq2_xxs emitter's DEFAULT m2
// integer-core anchor (the VLEN-universal floor, the byte-exact CORE target); the
// gearbox's VLEN256 m1 refinement is a separate schedule-pass concern NOT exercised
// here. COVERAGE at the default anchor.
//
// iq2_xxs FLIP (L3 coverage, SIGN-PLANE signs64): the front door no longer constructs
// the retired monolith op -- it constructs the typed super-block SCALAR-accumulator GRID
// loop body (fold_model "scalar_delta_grid", stride 66) with the iq2_xxs GRID-of-8
// grid-core brick (weft_rvv.iq2_xxs_q8_k_grid_core). The export still resolves through the
// SAME super-block monolithic route family (kind/ABI/facts) by the selector +
// weight_block_stride 66, so the emission-plan metadata + the exported object are
// byte-unchanged, and the CORE EmitC is byte-identical to the retired monolith modulo the
// source-op provenance token. The constructed brick is left attr-less at construction
// (default m2 anchor = the byte-exact CORE target); its Win-A m2/m1 gearbox is PRESERVED
// on the brick (kernel key "iq2_xxs") and refined by the schedule pass, a separate concern
// NOT exercised here (see rvv-iq2-xxs-q8-k-block-dot-autotuner-divergence.mlir).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block grid-codebook
// block-dot body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door --weft-rvv-materialize-schedule=march=rv64gcv --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door --weft-rvv-materialize-schedule=march=rv64gcv --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door --weft-rvv-materialize-schedule=march=rv64gcv --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door --weft-rvv-materialize-schedule=march=rv64gcv --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the iq2_xxs GRID-codebook gather (each weight byte
// INDEXES the packed uint64 grid via an indexed vluxei16 gather -- NOT an arithmetic
// decode -- with the per-element sign read from a DERIVED signs64 table and folded
// onto the GRID), wrapped in the super-block scale machinery -- pinned so a
// regression into an arithmetic decode, a wrong sign fold, or a wrong scale domain
// is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_iq2_xxs_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_iq2_xxs_q8_K_kernel"} {
  func.func @source_iq2_xxs_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_iq2_xxs_q8_K_kernel
// iq2_xxs FLIP (L3 coverage, SIGN-PLANE signs64): the front door now constructs the typed
// super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride
// 66) with the iq2_xxs GRID-of-8 grid-core brick, NOT the retired monolith op. The export
// still resolves through the SAME super-block monolithic route family (kind/ABI/facts) by
// the selector + weight_block_stride 66, so the emission-plan metadata is byte-unchanged.
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq2_xxs kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq2_xxs_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq2_xxs_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC grid-codebook integer core =================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot
// The GRID codebook emitted as a structured static const int64_t[256] decl (ggml's
// exact packed uint64 iq2xxs_grid literals), read via a (const int8_t *) cast.
// CORE: verbatim "static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL,
// The DERIVED signs64 sign table (= keven_signs_q2xs +-1, expanded in-emitter from
// the ksigns selector plane) emitted as a structured static const int8_t[1024] decl.
// CORE: verbatim "static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1,
// The super-block count nb = n / 256.
// CORE: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// aux1 reassembled from little-endian byte loads (alignment-safe, no *(uint32_t*)).
// CORE: bitwise_left_shift
// CORE: bitwise_or
// ls = 2*(aux1>>28)+1.
// CORE: bitwise_right_shift
// The vluxei16 grid+sign gather chain, BATCHED per sub-block PAIR (2*core = m4): the
// 8-slot u16 byte-offset index load (u16m1 EMUL), the i64m4 GRID gather reinterpreted to
// i8m4, the i64m4 DERIVED-signs gather, the i8m4 q8 pair load, then the +-1 signs folded
// onto the GRID (NOT q8) by vmul_vv_i8m4.
// CORE: call_opaque "__riscv_vle16_v_u16m1"
// CORE: call_opaque "__riscv_vluxei16_v_i64m4"
// CORE: call_opaque "__riscv_vreinterpret_v_i64m4_i8m4"
// CORE: call_opaque "__riscv_vluxei16_v_i64m4"
// CORE: call_opaque "__riscv_vle8_v_i8m4"
// CORE: call_opaque "__riscv_vmul_vv_i8m4"
// Each 32-lane sub-block half is recovered by a register-group vget i8m4->i8m2, then the
// SAME signed widening product + ONE vwredsum per sub-block + scalar extract.
// CORE: call_opaque "__riscv_vget_v_i8m4_i8m2"
// CORE: call_opaque "__riscv_vwmul_vv_i16m4"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// bsum += sumi*ls (integer domain).
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fold: ONE expression d*(float)bsum then sumf + that.
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The trailing 0.125f factor + the *s store.
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot
