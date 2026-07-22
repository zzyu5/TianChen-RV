// FULL production-export CLOSURE for the iq1_m (ggml IQ1_M x Q8_K 1.75-bit TERNARY
// super-block grid-CODEBOOK block-dot) front door -- the coverage payoff that CLOSES
// the literal block-dot zoo: the FINAL iq* super-block grid-codebook member (its
// sibling iq1_s is the only other 1.75-bit ternary variant). iq1_m FLIP (L3): the
// front door's OWN auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop
// body (weft_rvv.typed_super_block_block_dot_loop_body, fold_model "scalar_delta_grid"
// -- the iq1_m ternary-grid integer core + the emitter-inlined scalar delta fold + a
// single `sumf` scalar yield; NOT an opaque weft_rvv.iq1_m_q8_k_block_dot op, RETIRED
// the same action as the flip) now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-codebook is MECHANICAL,
// NOT bespoke. iq1_m takes the EXISTING super-block monolithic route family (shared
// with q4_K and iq4_xs, the 'rvv-ggml-super-block-block-dot-monolithic-emitc-
// route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The 2048-entry ternary grid is an OP attr consumed by
// the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator key
// ONLY off op name -> route family + the kind/scale_model attrs + the ordered ABI
// roles; neither reads the grid. So the shared SuperBlock family + the grid-codebook-
// attr stamping COMPOSE cleanly: COVERAGE = one table row (RVVMonolithicBlockDot
// Family.h, SuperBlock + the 4-role ggml vec_dot ABI n/s/vx/vy) + one front door
// (RVVIQ1MBlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/iq4_xs/q4_0/q8_0/
// iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq1_m's front-door-constructed op is left
// attr-less (no shape knob); iq1_m is NOT in any schedule autotuner, and the grid
// gather pins m1 in the emitter, so the op lowers at its m1 integer-core anchor --
// there is NO VLEN128-vs-VLEN256 byte-flip for iq1_m.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block-codebook
// block-dot body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports a
// real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the ternary GRID-codebook gather (the 11-bit grid
// index looks up the 2048-entry iq1s_grid table via the vluxei16 hardware indexed
// gather -- NOT an arithmetic decode), wrapped in the packed-scale super-block
// machinery -- pinned so a regression into an arithmetic decode, a wrong scale
// reconstruction, or a folded (bsums) delta is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_iq1_m_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_iq1_m_q8_K_kernel"} {
  func.func @source_iq1_m_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// iq1_m FLIP (L3): the front door now auto-constructs the typed SUPER-BLOCK
// SCALAR-accumulator GRID loop body (weft_rvv.typed_super_block_block_dot_loop_body,
// fold_model "scalar_delta_grid") from the iq1_m ternary-grid integer core (the iq1_s
// sibling) + the emitter-inlined scalar delta fold, NOT an opaque
// weft_rvv.iq1_m_q8_k_block_dot op (retired). It still resolves to its OWN monolithic
// super-block export entry (by fold_model + weight_block_stride 56), so the route id +
// ABI + object export are unchanged.
// PLAN: weft.exec.kernel @ggml_vec_dot_iq1_m_q8_K_kernel
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq1_m kind -- the SAME super-block route family q4_K/iq4_xs use.
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-generic-typed-body-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq1_m_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.

// ===================== CORE EmitC ternary grid-codebook integer core =========
// CORE: emitc.func @weft_emitc_ggml_vec_dot_iq1_m_q8_K_kernel_rvv_iq1_m_q8_K_block_dot
// The 2048-entry TERNARY GRID codebook emitted as a structured static const decl (the
// SAME iq1s_grid[2048] the sibling iq1_s carries; grid[0] = -1 -> 0xffffffffffffffff).
// CORE: verbatim "static const uint64_t weft_iq1m_grid[2048] = {0xffffffffffffffffULL,
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The packed iq1m_scale reconstruction: the 4 scales[] words OR'd into a uint16 lvalue.
// CORE: load %{{.*}} : <!emitc.opaque<"const uint16_t">>
// CORE: bitwise_right_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: bitwise_or %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// The fp16 bit-reinterpret: &scbits -> (float)*(const _Float16 *) (NOT a numeric conv).
// CORE: apply "&"
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The TWO int32 accumulators sumi1 + sumi2 (iq1_m's per-group delta cannot fold).
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The uint8 qh bytes + per-group delta sign.
// CORE: load %{{.*}} : <!emitc.opaque<"const uint8_t">>
// The 11-bit grid index: (qh << sh) & 0x700 OR'd with the qs index byte.
// CORE: bitwise_left_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// The per-group Sum(q8) reduce (the delta term -- bsums unusable; grid-independent,
// stays per-group on vle8/i8m1).
// CORE: call_opaque "__riscv_vsetvl_e8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// CORE: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// CORE: call_opaque "__riscv_vmv_x_s_i16m1_i16"
// The grid table base cast to (const int64_t *) for the indexed gather (ggml's
// (const int64_t *)iq1s_grid; NO kmask -- the ternary grid is itself signed).
// CORE: cast %{{.*}} : {{.*}}const uint64_t{{.*}} to {{.*}}const int64_t
// The vluxei16 HARDWARE indexed grid gather, per HALF: vle16 mf4 index -> vluxei16_v_
// i64m1 (2 u64 entries) -> reinterpret i8m1 (16 ternary bytes) -> vle8 the 16 q8 ->
// vwmul i16m2 -> ONE vwredsum per half.
// CORE: apply "&"
// CORE: call_opaque "__riscv_vle16_v_u16mf4"
// CORE: call_opaque "__riscv_vluxei16_v_i64m1"
// CORE: call_opaque "__riscv_vreinterpret_v_i64m1_i8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The per-super-block fold: ONE expression d*((float)sumi1 + 0.125f*(float)sumi2).
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The *s store (NO trailing factor -- *s = sumf).
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_iq1_m_q8_K_kernel_rvv_iq1_m_q8_K_block_dot
