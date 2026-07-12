// FULL production-export CLOSURE for the tq1_0 (ggml TQ1_0 x Q8_K super-block
// TERNARY block-dot) front door -- the LITERAL LAST of the 24 ggml dot kernels
// (100% front-door coverage). The front door's OWN auto-constructed monolithic
// super-block-ternary block-dot body now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-ternary is
// MECHANICAL, NOT bespoke. tq1_0 takes the EXISTING super-block monolithic route
// family (shared with q4_K / iq4_xs / the plain K-quants -- the
// 'rvv-ggml-super-block-block-dot-monolithic-emitc-route-family' route id + the
// super-block op-derived metadata keys), NOT a new route-id / family variant. The
// base-3 trit unpack (both qs and qh weight arrays), the flat-256 integer
// accumulator, and the single-scale scalar fp32 fold are OP structure consumed by
// the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator
// key ONLY off op name -> route family + the kind/scale_model attrs + the ordered
// ABI roles; neither reads the ternary decode. So the shared SuperBlock family +
// the tq1_0 attr stamping COMPOSE cleanly: COVERAGE = one table row
// (RVVMonolithicBlockDotFamily.h, SuperBlock + the 4-role ggml vec_dot ABI
// n/s/vx/vy) + one front door (RVVTQ10BlockDotSourceFrontDoor.cpp), NOT any new
// mechanism. q4_K/q4_0/q8_0/iq4_nl/iq4_xs stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. tq1_0's front-door-constructed op is
// left attr-less (no shape knob); tq1_0 IS in a schedule autotuner (kernel key
// "tq1_0"), but the front door leaves the integer_core_lmul knob unstamped, so the
// op lowers at its DEFAULT m2 integer-core anchor (the VLEN-universal-safe floor,
// byte-exact at VLEN128) -- stamping the m1 VLEN256 anchor is a separate
// schedule-descriptor concern, not this coverage closure.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block-ternary
// block-dot body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the tq1_0 BASE-3 TERNARY decode (each trit
// recovered by the mandatory uint8-wrap `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q
// *3)>>8; xi-1`) into aux8[256], a SINGLE flat-256 integer accumulator, and a
// SINGLE-SCALE SCALAR fp32 fold -- pinned so a regression into a 2-bit-field decode,
// a nibble/min K-quant, or a wrong scale domain is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_tq1_0_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_tq1_0_q8_K_kernel"} {
  func.func @source_tq1_0_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_tq1_0_q8_K_kernel
// tq1_0 FLIP (C_construct 25->26, the SECOND TQ-family member): the front door now
// constructs the typed super-block SCALAR-accumulator BASE-3 TERNARY loop body (fold_model
// "scalar_delta_grid", stride 54) with the tq1_0 BASE-3 ternary integer-core brick, NOT the
// retired monolith op. It REUSES the tq2_0 ternary scaffold at C2 marginal cost. The export
// still resolves through the SAME super-block monolithic route family (kind/ABI/facts) by the
// selector, so the emission-plan metadata is byte-unchanged and the CORE EmitC is
// byte-identical to the retired monolith modulo the source-op provenance token.
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the tq1_0 kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_tq1_0_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_tq1_0_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC base-3 ternary integer core ================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot
// The int8_t aux8[256] scratch + the scalar sumf, zeroed ONCE outside the loop.
// CORE: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The BASE-3 trit unpack (the load-bearing decode `q=(uint8_t)(byte*pow3[l]); xi=
// ((uint16_t)q*3)>>8; xi-1`): the u8m2 load of the qs chunk, the uint8 wrap vmul.vx
// by pow3[l] (NOT widened -- the mod-256 truncation IS the decode), the widening
// vwmulu by 3, the vsrl by 8, the u16->u8 narrow, the u8->i8 reinterpret, then the
// per-element `-1` ternary bias (vadd.vx of -1) BEFORE the vse8.
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vmul_vx_u8m2"
// CORE: call_opaque "__riscv_vwmulu_vx_u16m4"
// CORE: call_opaque "__riscv_vsrl_vx_u16m4"
// CORE: call_opaque "__riscv_vncvt_x_x_w_u8m2"
// CORE: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CORE: call_opaque "__riscv_vadd_vx_i8m2"
// CORE: call_opaque "__riscv_vse8_v_i8m2"
// tq1_0 is BASE-3, NOT a 2-bit field shift (tq2_0) and NOT a nibble/min K-quant: no
// `& 3` 2-bit field mask, no 4-bit nibble scale extraction between the trit unpack
// and the dot.
// CORE-NOT: call_opaque "__riscv_vand_vx_u8m2"
// CORE-NOT: bitwise_right_shift
// The SINGLE per-super-block integer accumulator sumi, fed by the WIDE-strip
// flat-256 dot over aux8 x q8 at the default m2 anchor: vle8 i8m2 -> vwmul_vv_i16m4
// -> vwredsum_vs_i16m4_i32m1 -> vmv_x_s (NO per-sub-block scale multiply -- tq1_0
// has no scales).
// CORE: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vwmul_vv_i16m4"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The OLD narrow dot path (per-16-lane vwmul_i16m2 reduce) is gone.
// CORE-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// The SINGLE-SCALE SCALAR fp32 fold: dx via ONE fp16 read seam (at xb+52, the END
// of block_tq1_0), dy (fp32 q8_K scale) loaded once, then `sumf += (float)sumi *
// (dx*dy)` -- a cast + a mul + an add. Exactly ONE fp16 read (no dall/dmin pair).
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: expression
// CORE: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// tq1_0 has NO min term (no dall/dmin pair, no bsums subtract): NO float subtract
// between the fold add and the RVV variant's *s store + return.
// CORE-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store, then return (bounds the min-term guard to the RVV variant body).
// CORE: return

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot
