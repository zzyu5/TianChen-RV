// FULL production-export CLOSURE for the nvfp4 (ggml NVFP4 x Q8_0 FP4-CODEBOOK
// block-dot) front door -- the sibling that CLOSES the literal block-dot zoo (the
// SECOND FP4-codebook op, after iq4_nl's non-linear int8 table). The front door's OWN
// auto-constructed typed FLAT block-dot loop body (fold_model "flat_nvfp4_codebook",
// the nvfp4 CODEBOOK integer-core brick -- the flip retired the monolith op) flows
// through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): nvfp4 is MECHANICAL, NOT bespoke.
// Its FLAT block_q8_0 activation (stride 34) puts it on the EXISTING flat monolithic
// route family (the 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family' route id
// + the flat op-derived metadata keys), the SAME route q4_0/iq4_nl use, NOT a new
// route-id / family variant. The 16-entry FP4 (e2m1) DOUBLED codebook and the UE4M3
// per-sub-block weight scale are OP attrs consumed by the emitter, NOT route-family
// concerns: the emission plan (buildMonolithicBlockDotEmissionPlan) and the
// target-export candidate validator key ONLY off op name -> route family + the
// kind/scale_model attrs + the ordered ABI roles; neither reads the codebook or the
// scale decode. So the shared Flat family + nvfp4's codebook/UE4M3 stamping COMPOSE
// cleanly: COVERAGE = one table row (RVVMonolithicBlockDotFamily.h, Flat + the 4-role
// ggml vec_dot ABI n/s/vx/vy) + one front door (RVVNVFP4BlockDotSourceFrontDoor.cpp),
// NOT any new mechanism. q4_0/q8_0/iq4_nl/q4_K/iq4_xs stay byte-exact on their routes.
//
// NO SCHEDULE PASS -- the codebook gather pins the m1 anchor in the emitter (the
// inversion vs iq4_nl). Unlike iq4_nl (a TunableScheduleOpInterface op whose attr-less
// op is shaped by --weft-rvv-materialize-schedule and rides its m1<->mf2 VLEN flip),
// nvfp4 is NOT tunable: the codebook gather admits ONLY the m1 anchor (mf2's VLMAX < 16
// at VLEN=128 cannot host the 16-entry gather), so there is NO nvfp4 schedule autotuner
// and NO VLEN128-vs-VLEN256 byte-flip. The nvfp4 codebook-core brick carries NO
// integer_core_lmul shape knob (the emitter's fixed vrgather/i8m1/i16m2 codebook dot is
// the m1 anchor) -- exactly the shape the constructed CORE emit fixture
// (rvv-to-emitc-nvfp4-q8-0-typed-flat-block-dot-loop-body) pins. So NO
// --weft-rvv-materialize-schedule appears in any RUN line below.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the typed FLAT FP4-codebook block-dot
// loop body (the codebook gather pins m1 in the emitter -- no schedule pass), the
// weft-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --weft-check-execution-plan-coherence (the flat monolithic route id is a registered
// target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export EmitC
// is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the FP4 CODEBOOK gather (the 4-bit nibble INDEXES
// the 16-entry doubled e2m1 table via vrgather -- NOT a linear nibble-8 decode),
// wrapped in the nvfp4 super-block UE4M3-scale machinery -- pinned so a regression
// into an arithmetic decode, a wrong scale domain (mxfp4's E8M0 bit dance), or a
// missing *0.5f half-form is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_nvfp4_q8_0_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_nvfp4_q8_0_kernel"} {
  func.func @source_nvfp4_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic emission-plan
// diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_nvfp4_q8_0_kernel
// nvfp4 FLIP (C_construct 27->28, the LAST dispatch-wired vec_dot): the front door
// now constructs the typed FLAT block-dot loop body (fold_model
// "flat_nvfp4_codebook", stride 36) with the nvfp4 CODEBOOK integer-core brick
// (mxfp4's 16-entry vrgather e2m1 codebook gather + the per-sub-block UE4M3 fp8
// weight scale), NOT the retired monolith op. The export still resolves through the
// SAME flat monolithic route family (kind/ABI/facts) by the selector +
// fold_model "flat_nvfp4_codebook", so the emission-plan metadata is byte-unchanged.
// PLAN: weft_rvv.typed_flat_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys,
// with the nvfp4 kind -- the SAME flat route family q4_0/iq4_nl use.
// PLAN-SAME: rvv_ggml_flat_block_dot_kind
// PLAN-SAME: ggml_nvfp4_q8_0_block_dot
// The honest FLAT monolithic-body route id (NOT q4_K/iq4_xs's super-block route, NOT
// the decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-flat-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_nvfp4_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC FP4-codebook integer core ==================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot
// The FP4 codebook emitted as a structured static const int8_t[16] decl (the SAME
// doubled e2m1 lookup table kvalues_mxfp4[16] mxfp4 uses).
// CORE: verbatim "static const int8_t weft_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The function-scoped fp32 accumulator + the SUPER-block count nb = n / 64.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the super-block loop), i8m1.
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The STRUCTURED UE4M3 -> fp32 weight scale (the genuinely-new NVFP4-class piece): the
// exp/man split, the two ldexpf branches, the exp==0 conditional, the *0.5f, then the
// two specials (e==0 || e==0x7F -> 0.0f).
// CORE: bitwise_right_shift %{{.*}}, %{{.*}}
// CORE: bitwise_and %{{.*}}, %{{.*}}
// CORE: bitwise_and %{{.*}}, %{{.*}}
// CORE: call_opaque "ldexpf"
// CORE: call_opaque "ldexpf"
// CORE: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// CORE: logical_or %{{.*}}, %{{.*}}
// CORE: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// mxfp4's E8M0 bit-dance MUST be absent (this is the UE4M3 ldexpf path, not a misroute).
// CORE-NOT: 0x00200000
// The q8_0 fp16->fp32 scale read (the OTHER sanctioned opaque scalar piece).
// CORE: call_opaque "(float)*(const _Float16 *)"
// The 8-lane sub-block setvl + the packed FP4 weight u8 load + the two q8 signed halves.
// CORE: call_opaque "__riscv_vsetvl_e8m1"
// CORE: call_opaque "__riscv_vle8_v_u8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The CODEBOOK class mechanism: split the nibble into the two UNSIGNED index lanes,
// then GATHER each through the broadcast table.
// CORE: call_opaque "__riscv_vand_vx_u8m1"
// CORE: call_opaque "__riscv_vsrl_vx_u8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook, not nibble-8).
// CORE-NOT: call_opaque "__riscv_vxor_vx_i8
// The shared asymmetric widening product (i8m1 x i8m1 -> i16m2): low then + high.
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwmacc_vv_i16m2"
// The vwredsum into i32m1 + the scalar extract.
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The nvfp4 fold: dy * d FIRST, then * (float)sumi, then sumf + that.
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot
