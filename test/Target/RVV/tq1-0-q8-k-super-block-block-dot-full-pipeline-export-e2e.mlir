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
// 'rvv-generic-typed-body-emitc-route-family' route id + the
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

// The CORE EmitC integer core is the FUSED tq1_0 BASE-3 TERNARY vec_dot leaf (the
// P1-proven owned VLEN-universal structure): each trit recovered by the mandatory
// uint8-wrap `xi=((uint16_t)byte*3)>>8; (u16)(xi-1) reinterpret i16`, q8 PRE-WIDENED
// to i16, folded into a SINGLE i16m4 accumulator (vmul_vv init / vmacc chain, NO
// aux8[256] scratch store/reload), reduced by ONE vwredsum, then a SINGLE-SCALE
// SCALAR fp32 fold -- pinned so a regression into a 2-bit-field decode, a nibble/min
// K-quant, a wrong scale domain, OR the retired aux8 + 8x-serial-vwredsum form is
// caught. --implicit-check-not forbids the retired ops anywhere in the emit.
// RUN: FileCheck %s --check-prefix=CORE --implicit-check-not=aux8 --implicit-check-not=vse8_v_i8m2 --implicit-check-not=vncvt_x_x_w_u8m2 --implicit-check-not=vwmul_vv_i16m4 --implicit-check-not=vwmul_vv_i16m2 --implicit-check-not=vadd_vx_i8m2 < %t.core.mlir

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
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-generic-typed-body-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_tq1_0_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.

// ===================== CORE EmitC base-3 ternary FUSED integer core ==========
// CORE: emitc.func @weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot
// The qh SINGLE-pass plane-weight table (function scope) + the scalar sumf, zeroed
// ONCE outside the loop. NO aux8[256] scratch: the fused leaf keeps the WHOLE
// super-block dot in ONE i16m4 accumulator (no store/reload round-trip).
// CORE: static const uint8_t weft_tq1_0_pow16[16]
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The FUSED base-3 trit decode (`xi=((uint16_t)byte*3)>>8; (u16)(xi-1) reinterpret
// i16`) with q8 PRE-WIDENED to i16 (vwcvt), paired directly into the accumulator:
// digit 0 of main qs is the vmul_vv INIT. The u8 wrap vmul.vx by pow3[l] feeds the
// widening vwmulu by 3, the vsrl by 8, the (u16 - 1) vsub, then the u16->i16
// reinterpret -- NO u16->u8 narrow, NO aux8 store.
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vwmulu_vx_u16m4"
// CORE: call_opaque "__riscv_vsrl_vx_u16m4"
// CORE: call_opaque "__riscv_vsub_vx_u16m4"
// CORE: call_opaque "__riscv_vreinterpret_v_u16m4_i16m4"
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vwcvt_x_x_v_i16m4"
// CORE: call_opaque "__riscv_vmul_vv_i16m4"
// digits 1..4 of main qs: the pow3[l] u8 wrap (vmul.vx) then the PLAIN vmacc (vl=32,
// no tail to preserve) accumulate into the SAME i16m4 accumulator (init + 4 macc).
// CORE: call_opaque "__riscv_vmul_vx_u8m2"
// CORE: call_opaque "__riscv_vmacc_vv_i16m4"
// The tail qs (5 macc) uses the tail-UNDISTURBED vmacc `_tu` at vl=16: it MUST keep
// the accumulator's upper lanes 16..31 (the main contributions the vl=32 reduce
// sums), which the tail-agnostic default may clobber -- byte-exact + VLEN-universal.
// CORE: call_opaque "__riscv_vmacc_vv_i16m4_tu"
// tq1_0 is BASE-3, NOT a 2-bit field shift (tq2_0) and NOT a nibble/min K-quant: no
// `& 3` 2-bit field mask, no 4-bit nibble scale extraction.
// CORE-NOT: call_opaque "__riscv_vand_vx_u8m2"
// The qh SINGLE pass: the 4 qh bytes read as a little-endian u32, broadcast x4 via
// vmv.v.x u32m2 + reinterpret to u8m2, multiplied lane-wise by the pow16 plane
// weights (vmul_vv_u8m2), then the SAME trit decode + ONE tail-undisturbed vmacc
// (NOT 4 per-plane aux8 stores).
// CORE: call_opaque "__riscv_vmv_v_x_u32m2"
// CORE: call_opaque "__riscv_vreinterpret_v_u32m2_u8m2"
// CORE: call_opaque "__riscv_vmul_vv_u8m2"
// CORE: call_opaque "__riscv_vmacc_vv_i16m4_tu"
// The SINGLE reduce: ONE vwredsum over exactly 32 active lanes (fixed vl ->
// VLEN-universal) folds the whole i16m4 accumulator into sumi -- NOT the retired
// 8x serial per-strip vwredsum chain.
// CORE: call_opaque "__riscv_vmv_v_x_i32m1"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
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
