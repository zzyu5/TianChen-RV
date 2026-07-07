// FULL production-export CLOSURE for the iq3_xxs (ggml IQ3_XXS x Q8_K super-block
// GRID-CODEBOOK block-dot) front door -- the coverage payoff that CLOSES THE LITERAL
// BLOCK-DOT ZOO (the final iq* super-block-codebook bucket). iq3_xxs is a GRID-of-4
// codebook super-block quant: a 256-entry uint32 grid (each entry 4 int8 values) +
// the 128-entry ksigns_iq2xs sign plane wrapped in q4_K's super-block structure. The
// front door's OWN auto-constructed monolithic super-block-grid-codebook body now
// flows through the COMPLETE tcrv-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --tcrv-check-execution-plan-coherence) AND
// exports a real RISC-V target artifact through tcrv-translate
// --tcrv-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-grid-codebook is
// MECHANICAL, NOT bespoke. iq3_xxs takes the EXISTING super-block monolithic route
// family (shared with q4_K, the 'rvv-ggml-super-block-block-dot-monolithic-emitc-
// route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The grid + ksigns tables are OP attrs consumed by the
// emitter, NOT a route-family concern: the emission plan (buildMonolithicBlockDot
// EmissionPlan) and the target-export candidate validator key ONLY off op name ->
// route family + the kind/scale_model attrs + the ordered ABI roles; neither reads
// the grid. So the shared SuperBlock family + iq3_xxs's grid/ksigns-attr stamping
// COMPOSE cleanly: COVERAGE = one table row (RVVMonolithicBlockDotFamily.h,
// SuperBlock + the 4-role ggml vec_dot ABI n/s/vx/vy) + one front door
// (RVVIQ3XXSBlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/iq4_xs and the
// rest stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq3_xxs's front-door-constructed op is
// left attr-less (no shape knob); iq3_xxs is NOT in any schedule autotuner, and the
// 1024-byte grid does NOT broadcast into a vreg (the grid-of-4 lookup is a hardware
// vluxei16 indexed gather), so the op lowers at its VLEN>=128 m1 integer-core anchor
// -- there is NO VLEN128-vs-VLEN256 byte-flip for iq3_xxs.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block-grid-codebook
// block-dot body, the tcrv-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --tcrv-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the iq3 GRID-of-4 codebook gather (the weight byte
// INDEXES the 256-entry uint32 grid via a hardware vluxei16 indexed gather -- NOT a
// vrgather broadcast, the 1024-byte table cannot broadcast into a vreg), the ksigns
// sign plane, and the int-domain scale fold -- pinned so a regression into a wrong
// gather / sign / scale domain is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_iq3_xxs_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_iq3_xxs_q8_K_kernel"} {
  func.func @source_iq3_xxs_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_iq3_xxs_q8_K_kernel
// iq3_xxs FLIP (L3 coverage): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 98) with
// the iq3_xxs GRID-of-4 grid-core brick, NOT the retired monolith op. The export still
// resolves through the SAME super-block monolithic route family (kind/ABI/facts) by the
// selector + weight_block_stride 98, so the emission-plan metadata is byte-unchanged.
// PLAN: tcrv_rvv.typed_super_block_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq3_xxs kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq3_xxs_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq3_xxs_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC grid-codebook integer core =================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot
// The GRID-of-4 codebook + sign plane emitted as structured static const decls (the
// grid as a uint32_t[256] rendering ggml's exact literals, read via a (const int8_t
// *) cast; ksigns as a uint8_t[128]; kmask as the inline {1<<j} const).
// CORE: verbatim "static const uint32_t tcrv_iq3xxs_grid[256] = {0x04040404U,
// CORE: verbatim "static const uint8_t tcrv_iq3xxs_ksigns[128] = {0, 129, 130, 3,
// CORE: verbatim "static const uint8_t tcrv_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The FULL 8-bit kmask broadcast-loaded ONCE (above the super-block loop), u8m1.
// CORE: call_opaque "__riscv_vle8_v_u8m1"
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// aux32 reassembled from 4 little-endian byte loads (alignment-safe, no *(uint32_t*)).
// CORE: bitwise_left_shift
// CORE: bitwise_or
// ls = 2*(aux32>>28)+1.
// CORE: bitwise_right_shift
// The GRID-of-4 gather, BATCHED per SUB-BLOCK PAIR (the AVL=2 fractional-LMUL fix): the
// 32 per-group vl=2 vluxei16_v_i32m1 gathers (each with a vl=2 u16mf2 index load) are
// hoisted to ONE wide gather per pair -- a uint16_t[32] byte-offset array (4 slots/group:
// 2 real idx*4 + 2 zero pads) vle16'd at u16m4, gathered by ONE vluxei16_v_i32m8, and
// reinterpreted to i8m8. Each group's 8 grid bytes are recovered in lanes 0..7 by a
// register-group vget (i8m8 -> i8m1), then the 8-lane q8 load + the UNCHANGED sign fold.
// CORE: "emitc.variable"() {{.*}} -> !emitc.array<32x!emitc.opaque<"uint16_t">>
// CORE: call_opaque "__riscv_vle16_v_u16m4"
// CORE: call_opaque "__riscv_vluxei16_v_i32m8"
// CORE: call_opaque "__riscv_vreinterpret_v_i32m8_i8m8"
// CORE: call_opaque "__riscv_vget_v_i8m8_i8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The SIGN plane: broadcast signs / vand kmask / vmsne / vneg / vmerge.
// CORE: call_opaque "__riscv_vmv_v_x_u8m1"
// CORE: call_opaque "__riscv_vand_vv_u8m1"
// CORE: call_opaque "__riscv_vmsne_vx_u8m1_b8"
// CORE: call_opaque "__riscv_vneg_v_i8m1"
// CORE: call_opaque "__riscv_vmerge_vvm_i8m1"
// The signed widening product + the chained vwredsum + scalar extract.
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// bsum += sumi*ls (integer domain).
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fold: ONE expression d*(float)bsum then sumf + that.
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The trailing 0.25f factor + the *s store.
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot
