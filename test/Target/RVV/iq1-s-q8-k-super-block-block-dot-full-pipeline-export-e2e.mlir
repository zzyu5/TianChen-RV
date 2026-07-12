// FULL production-export CLOSURE for the iq1_s (ggml IQ1_S x Q8_K super-block
// TERNARY-grid CODEBOOK block-dot) front door -- the coverage payoff that CLOSES
// the literal block-dot zoo: iq1_s is the LAST iq* super-block-codebook format to
// gain a front door (the TERNARY class, sibling of iq1_m). iq1_s FLIP (L3 M3): the
// front door's OWN auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop
// body (weft_rvv.typed_super_block_block_dot_loop_body, fold_model
// "scalar_delta_grid" -- the iq1_s ternary-grid integer core + the emitter-inlined
// scalar delta fold + a single `sumf` scalar yield; NOT an opaque
// weft_rvv.iq1_s_q8_k_block_dot op, RETIRED the same action as the flip) now flows
// through the COMPLETE weft-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --weft-check-execution-plan-coherence) AND exports
// a real RISC-V target artifact through weft-translate --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-codebook is
// MECHANICAL, NOT bespoke -- iq1_s composes onto the SAME super-block monolithic
// route family iq4_xs/q4_K use, despite its grid being the 2048-entry TERNARY
// iq1s_grid (a uint64 table gathered by vluxei16, NOT iq4_xs's 16-entry vrgather
// codebook). iq1_s takes the EXISTING super-block monolithic route family (the
// 'rvv-ggml-super-block-block-dot-monolithic-emitc-route-family' route id + the
// super-block op-derived metadata keys), NOT a new route-id / family variant. The
// grid is an OP attr consumed by the emitter, NOT a route-family concern: the
// emission plan (buildMonolithicBlockDotEmissionPlan) and the target-export
// candidate validator key ONLY off op name -> route family + the kind/scale_model
// attrs + the ordered ABI roles; neither reads the grid. So the shared SuperBlock
// family + the grid-attr stamping COMPOSE cleanly: COVERAGE = one table row
// (RVVMonolithicBlockDotFamily.h, SuperBlock + the 4-role ggml vec_dot ABI
// n/s/vx/vy) + one front door (RVVIQ1SBlockDotSourceFrontDoor.cpp), NOT any new
// mechanism. iq4_xs/q4_K/q4_0/q8_0/iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq1_s's front-door-constructed typed
// super-block GRID loop body carries no shape knob; iq1_s is NOT in any schedule
// autotuner, so it lowers at its fixed vluxei16/i8m2 grid-core anchor -- there is NO
// VLEN128-vs-VLEN256 byte-flip for iq1_s (VLEN128 and VLEN256 emit the SAME bytes).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the typed super-block SCALAR-accumulator
// GRID loop body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the TERNARY-grid indexed gather (the 11-bit index
// GATHERS packed uint64 grid entries via vluxei16 over a (const int64_t *) cast --
// the ternary grid bytes ARE the signed value, NO sign plane / vrgather codebook),
// wrapped in the super-block scale + delta-bsum machinery -- pinned so a regression
// into a sign-apply, a wrong index width, or a wrong scale/delta domain is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_iq1_s_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_iq1_s_q8_K_kernel"} {
  func.func @source_iq1_s_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// iq1_s FLIP (L3 M3): the front door now auto-constructs the typed SUPER-BLOCK
// SCALAR-accumulator GRID loop body (weft_rvv.typed_super_block_block_dot_loop_body,
// fold_model "scalar_delta_grid") from the iq1_s ternary-grid integer core + the
// emitter-inlined scalar delta fold, NOT an opaque weft_rvv.iq1_s_q8_k_block_dot op
// (retired). It still resolves to its OWN monolithic super-block export entry (by
// fold_model + weight_block_stride 50), so the route id + ABI + object export are
// unchanged.
// PLAN: weft.exec.kernel @ggml_vec_dot_iq1_s_q8_K_kernel
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq1_s kind -- the SAME super-block route family q4_K/iq4_xs use.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq1_s_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq1_s_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC ternary-grid integer core ==================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_rvv_iq1_s_q8_K_block_dot
// The 2048-entry TERNARY grid emitted as a structured static const uint64 decl
// (the SAME iq1s_grid[2048] ggml uses; 0xff bytes read as signed -1 via a
// (const int8_t *) cast -- NO sign-bit plane).
// CORE: verbatim "static const uint64_t weft_iq1s_grid[2048] = {0xffffffffffffffffULL,
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The TWO int32 accumulators sumi + sumi1 (kept separate, the delta mechanism).
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The uint16 qh word load -> scale ls = 2*((qh>>12)&7)+1 + delta = 1 - 2*((qh>>15)&1).
// CORE: load %{{.*}} : <!emitc.opaque<"const uint16_t">>
// CORE: bitwise_right_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// The 11-bit grid index: ((qh>>3l)&7)<<8 OR'd with the qs index byte, then <<3 (idx*8).
// CORE: bitwise_left_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: bitwise_or %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// The HARDWARE indexed grid gather: vle16 the byte-offset indices (mf2 EMUL), gather
// the 4 grid u64 entries via vluxei16 over (const int64_t *)grid (NOT a vrgather --
// the 16 KB table cannot broadcast into a vreg), reinterpret to 32 signed i8.
// CORE: call_opaque "__riscv_vle16_v_u16mf2"
// CORE: cast %{{.*}} : {{.*}}const uint64_t{{.*}} to {{.*}}const int64_t
// CORE: call_opaque "__riscv_vluxei16_v_i64m2"
// CORE: call_opaque "__riscv_vreinterpret_v_i64m2_i8m2"
// The full 32-lane activation load + signed widening product (NO sign apply -- the
// ternary grid bytes are ALREADY signed) + ONE vwredsum + scalar extract.
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vwmul_vv_i16m4"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The DELTA term: the int16 bsums load + ls*delta*(bsums[2ib]+bsums[2ib+1]).
// CORE: load %{{.*}} : <!emitc.opaque<"const int16_t">>
// The per-super-block fold: ONE expression d*((float)sumi + 0.125f*(float)sumi1).
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The *s store (NO trailing factor -- *s = sumf).
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_rvv_iq1_s_q8_K_block_dot
