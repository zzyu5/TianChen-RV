// FULL production-export CLOSURE for the tq2_0 (ggml TQ2_0 x Q8_K super-block
// TERNARY block-dot) front door -- the LAST literal block-dot in the zoo (the 2-bit
// TriLM ternary K-quant). The front door's OWN auto-constructed monolithic
// super-block ternary block-dot body now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// WHY tq2_0 is MECHANICAL, NOT bespoke: tq2_0 is a genuine super-block (QK_K == 256)
// so it reuses the EXISTING super-block monolithic route family (the shared
// 'rvv-generic-typed-body-emitc-route-family' route id + the
// super-block op-derived metadata keys), NOT a new route-id / family variant. The
// block-format delta -- 2-bit TERNARY weights (`((qs >> shift) & 3) - 1` over shifts
// {0,2,4,6}, the `-1` bias folded into the unpack), a SINGLE per-super-block integer
// accumulator (NO scales[16], NO per-sub-block scale, NO min, NO dmin, NO bsums),
// and a SINGLE-fp16-scale SCALAR fp32 fold (`sumf += (float)sumi * d`, d at the END
// of block_tq2_0) -- is op STRUCTURE the emitter consumes, NOT a route-family or ABI
// concern. Its 4-role ggml vec_dot ABI (n, s, vx, vy) matches q4_K's. So the shared
// SuperBlock family + tq2_0's block-format constants COMPOSE cleanly: COVERAGE = one
// table row (RVVMonolithicBlockDotFamily.h) + one front door
// (RVVTQ20BlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/q2_K/iq4_xs/
// q4_0/q8_0/iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. tq2_0's front-door-constructed op is
// left attr-less (no integer_core_lmul shape knob). UNLIKE q4_K/q2_K/iq4_xs, tq2_0
// IS in a schedule autotuner (m2 at VLEN128, m1 at VLEN256), so there IS a
// VLEN128-vs-VLEN256 flip -- but the attr-less op lowers at the tq2_0 emitter's
// DEFAULT m2 integer-core anchor (the VLEN-universal floor, the byte-exact CORE
// target); the gearbox's VLEN256 m1 refinement is a separate schedule-pass concern
// NOT exercised here. COVERAGE at the default anchor.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block ternary
// block-dot body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-tq2-0-q8-k-block-dot-source-front-door=march=rv64gcv --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-tq2-0-q8-k-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-tq2-0-q8-k-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-tq2-0-q8-k-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the tq2_0 FUSED 2-bit TERNARY dot (the `-1` bias
// folded into the unpack, vwmacc'd DIRECTLY against q8, no aux8 spill) + the
// SINGLE-SCALE SCALAR fp32 fold -- pinned so a regression into a q2_K/q4_K-style
// scale/min decode, an aux8 spill, or a wrong fold is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_tq2_0_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_tq2_0_q8_K_kernel"} {
  func.func @source_tq2_0_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_tq2_0_q8_K_kernel
// tq2_0 FLIP (C_construct 24->25, the FIRST TQ-family member): the front door now
// constructs the typed super-block SCALAR-accumulator TERNARY loop body (fold_model
// "scalar_delta_grid", stride 66) with the tq2_0 FUSED 2-bit ternary integer-core brick,
// NOT the retired monolith op. The export still resolves through the SAME super-block
// monolithic route family (kind/ABI/facts) by the selector, so the emission-plan metadata
// is byte-unchanged and the CORE EmitC is byte-identical to the retired monolith modulo the
// source-op provenance token.
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic {{.*}}artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the tq2_0 kind -- the SAME super-block route family q4_K uses.
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-generic-typed-body-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_tq2_0_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.

// ===================== CORE EmitC tq2_0 ternary integer core =================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot
// The super-block count nb = n / 256. The FUSED dot has NO aux8[256] scratch (the
// 2-bit unpack is consumed in-register by the per-plane vwmacc, never spilled).
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CORE-NOT: !emitc.array<256x!emitc.opaque<"int8_t">>
// The scalar sumf, zeroed ONCE outside the super-block loop.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The outer super-block loop.
// CORE: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The SINGLE per-super-block integer accumulator sumi (default anchor m2 at the
// VLEN-universal floor; the gearbox refines m2->m1 at VLEN256 via a separate pass).
// CORE: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// The FUSED 2-bit TERNARY dot (ggml's _vl128 lane structure): a wide i16m4 plane
// accumulator zeroed ONCE for the whole super-block, each 32-byte qs chunk at e8m2,
// then 4 planes each unpacking 32 ternary lanes (vsrl/vand over {0,2,4,6}, u8->i8
// reinterpret, the per-element `-1` bias via vsub in the i8 domain) and vwmacc'd
// DIRECTLY against the matching 32 q8 lanes -- the load-bearing decode
// `((qs>>shift)&3) - 1` fused into one widening MAC, NO vse8 spill.
// CORE: call_opaque "__riscv_vsetvl_e16m4"
// CORE: call_opaque "__riscv_vmv_v_x_i16m4"
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vand_vx_u8m2"
// CORE: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CORE: call_opaque "__riscv_vsub_vx_i8m2"
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vwmacc_vv_i16m4"
// tq2_0 has NO scales / min / bsums machinery -- it must NOT misroute to a q2_K
// or q4_K-style scale/min decode. NO 4-bit nibble scale extraction, NO 6-bit
// utmp/kmask bit-dance, NO bsums. NO aux8 spill: the OLD narrow path (vse8 +
// per-16-lane vwmul + vsetvl_e8m1) is gone.
// CORE-NOT: bitwise_and
// CORE-NOT: bitwise_right_shift
// CORE-NOT: bitwise_or
// CORE-NOT: const int16_t
// CORE-NOT: call_opaque "__riscv_vse8_v_i8m2"
// CORE-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// ONE wide widening reduce for the WHOLE super-block (i16m4 -> i32m1 -> scalar),
// summed into sumi (NO per-sub-block scale multiply -- tq2_0 has no scales).
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SINGLE-SCALE SCALAR fp32 fold: dy (fp32 q8_K scale) loaded once, dx via the
// fp16 read seam (ONE read, at xb+64), d = dy*dx, then `sumf += (float)sumi * d`
// as ONE emitc.expression (a cast + a mul + an add). There is exactly ONE fp16
// read (no dall/dmin pair), NO subtract (no min term), NO 8-lane vfcvt/vfmul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: expression
// CORE: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CORE-NOT: call_opaque "__riscv_vfmacc
// CORE-NOT: call_opaque "__riscv_vse32_v_f32m2"

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot
