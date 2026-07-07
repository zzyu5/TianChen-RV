// FULL production-export CLOSURE for the iq2_xs (ggml IQ2_XS x Q8_K super-block
// GRID-codebook block-dot) front door -- the coverage payoff that closes the LAST
// iq* member of the super-block-codebook bucket: the packed-GRID + separate-SIGN-
// plane super-block dot (the SIBLING of iq2_xxs). The front door's OWN
// auto-constructed monolithic super-block grid-codebook body now flows through the
// COMPLETE tcrv-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through tcrv-translate --tcrv-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block grid-codebook is
// MECHANICAL, NOT bespoke. iq2_xs takes the EXISTING super-block monolithic route
// family (shared with q4_K / iq4_xs, the 'rvv-ggml-super-block-block-dot-monolithic-
// emitc-route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The 512-entry grid + the 128-entry ksigns sign plane are
// OP attrs consumed by the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator key
// ONLY off op name -> route family + the kind/scale_model attrs + the ordered ABI
// roles; neither reads the grid. So the shared SuperBlock family + the grid/ksigns-
// attr stamping COMPOSE cleanly: COVERAGE = one table row (RVVMonolithicBlockDot
// Family.h, SuperBlock + the 4-role ggml vec_dot ABI n/s/vx/vy) + one front door
// (RVVIQ2XSBlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/q4_0/q8_0/
// iq4_nl/iq4_xs stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering the
// direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq2_xs's front-door-constructed op is
// left attr-less (no shape knob); iq2_xs is NOT in any schedule autotuner, and UNLIKE
// iq4_nl/iq4_xs's 16-entry vrgather codebook, the iq2_xs grid lookup is an INDEXED
// gather over a pointer (the 4096-byte grid cannot broadcast into a vreg), so the
// integer core does NOT pin an m1 table anchor -- the op lowers at the iq2_xs
// emitter's shape and there is NO VLEN128-vs-VLEN256 byte-flip for iq2_xs.
//
// iq2_xs FLIP (L3 coverage, SIGN-PLANE signs64, PER-HALF explicit scale): the front door no
// longer constructs the retired monolith op -- it constructs the typed super-block
// SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid", stride 74) with the
// iq2_xs per-half-scale grid-core brick (tcrv_rvv.iq2_xs_q8_k_grid_core). The export still
// resolves through the SAME super-block monolithic route family (kind/ABI/facts) by the
// selector + weight_block_stride 74, so the emission-plan metadata + the exported object are
// byte-unchanged, and the CORE EmitC is byte-identical to the retired monolith modulo the
// source-op provenance token. UNLIKE iq2_xxs the brick carries NO gearbox (fixed 16-lane
// per-half shape, not in any autotuner) -- the per-half body is i64m1 gather + i8m1 view +
// i16m2 widen at all VLEN.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block grid-codebook
// block-dot body, the tcrv-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --tcrv-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports a
// real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the iq2_xs GRID-codebook gather (the 9-bit index
// looks up a packed uint64 grid entry via an INDEXED vluxei16, the per-element sign is
// read from the DERIVED signs64 plane -- NOT a vrgather, NOT an arithmetic decode),
// wrapped in the super-block explicit-int4-scale machinery + the 0.125f factor --
// pinned so a regression into a vrgather codebook or a wrong scale domain is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_iq2_xs_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_iq2_xs_q8_K_kernel"} {
  func.func @source_iq2_xs_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_iq2_xs_q8_K_kernel
// iq2_xs FLIP (L3 coverage, SIGN-PLANE signs64, PER-HALF explicit scale): the front door
// now constructs the typed super-block SCALAR-accumulator GRID loop body (fold_model
// "scalar_delta_grid", stride 74) with the iq2_xs per-half-scale grid-core brick
// (tcrv_rvv.iq2_xs_q8_k_grid_core), NOT the retired monolith op. The export still resolves
// through the SAME super-block monolithic route family (kind/ABI/facts) by the selector +
// weight_block_stride 74, so the emission-plan metadata is byte-unchanged.
// PLAN: tcrv_rvv.typed_super_block_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq2_xs kind -- the SAME super-block route family q4_K / iq4_xs use.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq2_xs_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq2_xs_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC grid-codebook integer core =================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot
// The 512-entry GRID codebook + the DERIVED signs64 sign table emitted as structured
// static const decls (signs64 = keven_signs_q2xs, expanded in-emitter from the ksigns
// selector: byte b of selector j is (ksigns[j] & (1<<b)) ? -1 : +1 -- ksigns[0]=0 ->
// 8x +1, ksigns[1]=129 (bits 0,7) -> -1,+1,...,+1,-1).
// CORE: verbatim "static const int64_t tcrv_iq2xs_grid[512] = {0x0808080808080808ULL,
// CORE: verbatim "static const int8_t tcrv_iq2xs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1,
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The i64 grid + signs64 views set up ONCE (above the super-block loop).
// CORE: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const int8_t">> to !emitc.ptr<!emitc.opaque<"const int64_t">>
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// DELTA(c): explicit scale byte -> ls1 = 2*(sc&0xf)+1, ls2 = 2*(sc>>4)+1.
// CORE: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: bitwise_right_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// DELTA(b): uint16 word reassembled from 2 LE byte loads (alignment-safe).
// CORE: bitwise_left_shift
// CORE: bitwise_or
// DELTA(a): idx = w & 511 (9-bit index into the 512-grid).
// CORE: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">)
// The sign selector w >> 9 (logical shift in uint32).
// CORE: bitwise_right_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"uint32_t">, !emitc.opaque<"uint32_t">)
// The vluxei16 grid+sign gather chain, BATCHED per sub-block PAIR (2 sub-blocks = 4
// halves): the 8-slot u16 byte-offset index load (u16m1 EMUL), the i64m4 GRID gather
// reinterpreted to i8m4, the i64m4 DERIVED-signs gather, the i8m4 q8 pair load, then the
// +-1 signs folded onto the GRID by vmul_vv_i8m4.
// CORE: call_opaque "__riscv_vle16_v_u16m1"
// CORE: call_opaque "__riscv_vluxei16_v_i64m4"
// CORE: call_opaque "__riscv_vreinterpret_v_i64m4_i8m4"
// CORE: call_opaque "__riscv_vluxei16_v_i64m4"
// CORE: call_opaque "__riscv_vle8_v_i8m4"
// CORE: call_opaque "__riscv_vmul_vv_i8m4"
// Each 16-lane half is recovered by a register-group vget i8m4->i8m1, then the SAME signed
// widening product + ONE vwredsum per half + scalar extract.
// CORE: call_opaque "__riscv_vget_v_i8m4_i8m1"
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
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

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot
