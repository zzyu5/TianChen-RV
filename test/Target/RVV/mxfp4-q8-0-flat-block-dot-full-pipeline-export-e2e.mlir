// FULL production-export CLOSURE for the mxfp4 (ggml MXFP4 x Q8_0 FLAT block-dot)
// front door -- the FP4 (e2m1) sibling of the iq4_nl codebook flat op, closing the
// LAST literal block-dot format bucket flowing through production-export: FP4
// weights with an E8M0 shared-exponent block scale. The front door's OWN
// auto-constructed monolithic flat block-dot body flows through the COMPLETE
// tcrv-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through tcrv-translate --tcrv-export-target-artifact.
//
// SAME generalized monolithic wiring (RVVMonolithicBlockDotFamily.h), NOT a new
// mechanism: mxfp4 is a FLAT op, so it takes the flat route id
// 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family' and carries the 4-role
// ggml vec_dot ABI (n, s, vx, vy) -- the SAME route family + ABI as the iq4_nl
// codebook sibling. Two mxfp4 facts are OP structure the emitter consumes, NOT a
// route-family concern: (1) the block format (block_mxfp4 stride 17, the FP4 nibbles
// at weight offset +1 after the single E8M0 exponent byte, distinct from iq4_nl's
// +2), and (2) the E8M0 -> fp32 HALF weight scale (kvalues_mxfp4 = 2*E2M1). q4_K /
// q4_0 / q8_0 / iq4_nl stay byte-exact on their own routes.
//
// SCHEDULE STAMP: like the iq4_nl codebook sibling, the mxfp4 emitter requires a
// stamped integer-core shape (the codebook gather anchor is a VLEN-capability fact),
// so the front-door-constructed attr-less op is shaped by the EXISTING
// --tcrv-rvv-materialize-schedule gearbox before lowering -- the same gearbox its
// CORE emit fixtures use. This is the honest per-op pipeline, not new wiring.
//
// BOTH MARCH ANCHORS (the emitter differs): the codebook gather FLIPS m1@VLEN128 ->
// mf2@VLEN256 (a byte-different core -- vrgather_vv_i8m1 + i16m2 product vs
// vrgather_vv_i8mf2 + i16m1 product), so this e2e closes production-export at BOTH
// the rv64gcv (m1) and rv64gcv_zvl256b (mf2) anchors. The already-board-sealed flip
// (rvv-mxfp4-q8-0-block-dot-autotuner-divergence.mlir) rides the unmodified gearbox;
// this test proves each stamped form reaches a real RISC-V object.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC, so the
// production-export emit is byte-identical to the CORE == emission-plans emit
// (asserted below by diff, per anchor, both under the same schedule stamp). NO board
// / NO perf claim -- coverage/wiring maturity only.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic flat block-dot body, the
// schedule gearbox stamps the integer-core shape (rv64gcv -> m1), the
// tcrv-source-artifact-front-door-pipeline materializes the emission plan AND passes
// --tcrv-check-execution-plan-coherence.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT @ VLEN128 (m1): --tcrv-materialize-emission-plans only APPENDS the
// emission-plan diagnostic mirror; the block-dot body is untouched, so the
// production-export EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

// BYTE-EXACT @ VLEN256 (the mf2 FLIP): the SAME auto-constructed attr-less op stamps
// mf2 at rv64gcv_zvl256b and lowers to a byte-different codebook core; the
// emission-plan mirror is still byte-exact vs the CORE emit at THIS anchor.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc > %t.core256.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod256.mlir
// RUN: diff %t.core256.mlir %t.prod256.mlir
// RUN: FileCheck %s --check-prefix=CORE256 < %t.core256.mlir

// Target-artifact OBJECT export @ VLEN128 (m1): the flat monolithic emission plan
// exports a real RISC-V RVV relocatable object through the registered peer object
// exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// Target-artifact OBJECT export @ VLEN256 (mf2): the byte-different mf2 core ALSO
// packages to a real RISC-V RVV relocatable object under the same registered
// exporter and the same exported handoff symbol.
// RUN: rm -f %t256.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t256.o
// RUN: llvm-readobj -h %t256.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t256.o | FileCheck %s --check-prefix=SYMBOL

module attributes {tcrv_rvv.source_front_door = "ggml_mxfp4_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_mxfp4_q8_0_kernel"} {
  func.func @source_mxfp4_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the FLAT monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_mxfp4_q8_0_kernel
// PLAN: tcrv_rvv.mxfp4_q8_0_block_dot
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline), with the mxfp4
// kind.
// PLAN-SAME: rvv_ggml_flat_block_dot_kind
// PLAN-SAME: ggml_mxfp4_q8_0_block_dot
// The honest FLAT monolithic-body route id (NOT q4_K's super-block route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-flat-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_mxfp4_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata and
// never claims the super-block route or a decomposed route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC FP4 codebook core @ VLEN128 (m1) ===========
// The CORE EmitC is the FP4 CODEBOOK gather (the 4-bit nibble INDEXES the 16-entry
// non-linear int8 kvalues_mxfp4 table via vrgather -- NOT a linear nibble-8 decode),
// wrapped in the structured E8M0 -> fp32 HALF weight-scale reconstruction -- pinned
// so a regression into an arithmetic decode or a wrong scale domain is caught.
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_rvv_mxfp4_q8_0_block_dot(
// The FP4 codebook emitted as a structured static const int8_t[16] decl
// (kvalues_mxfp4 = 2*E2M1).
// CORE: verbatim "static const int8_t tcrv_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The codebook table broadcast-loaded ONCE (above the block loop), i8m1.
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The STRUCTURED E8M0 -> fp32 HALF weight scale (the FP4-class piece): cmp / mask /
// shift / conditional, then the float pointer pun (NO fp16 read for the weight).
// CORE: cmp lt
// CORE: bitwise_and %{{.*}}, %{{.*}}
// CORE: bitwise_left_shift %{{.*}}, %{{.*}}
// CORE: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"uint32_t">
// CORE: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint32_t">> to !emitc.ptr<!emitc.opaque<"const float">>
// The SINGLE surviving q8_0 fp16->fp32 scale read (the weight fp16 read is GONE).
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE-NOT: call_opaque "(float)*(const _Float16 *)"
// The CODEBOOK mechanism: split the nibble into the two UNSIGNED index lanes, then
// GATHER each through the broadcast table (NOT the q4_0 offset-binary decode).
// CORE: call_opaque "__riscv_vand_vx_u8m1"
// CORE: call_opaque "__riscv_vsrl_vx_u8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// CORE-NOT: call_opaque "__riscv_vxor_vx_i8
// The shared asymmetric widening product + carried-seed vwredsum + scalar extract.
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwmacc_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The mxfp4 fold: scale_x * d_y, then (float)sumi * that, then sumf + that.
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)

// =========== CORE EmitC FP4 codebook core @ VLEN256 (the mf2 FLIP) ===========
// The SAME front door, a DIFFERENT capability fact: the codebook core FLIPS to the
// mf2 anchor (the table load / nibble gather / strip vsetvl narrow to e8mf2 -- a FULL
// mf2 register at VLEN256, VLMAX 16 = the ggml _vl256 shape) and the widened product
// narrows one step to i16m1. The structured E8M0 half scale is UNCHANGED (it is
// scalar, outside the vector core).
// CORE256: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_rvv_mxfp4_q8_0_block_dot(
// CORE256: verbatim "static const int8_t tcrv_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The codebook table broadcast-loaded at the mf2 anchor (16 entries fill a full mf2
// register at VLEN256); the m1-form strip spellings must be ABSENT.
// CORE256: call_opaque "__riscv_vle8_v_i8mf2"
// CORE256-NOT: call_opaque "__riscv_vsetvl_e8m1"
// CORE256-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// The structured E8M0 -> fp32 HALF scale is UNCHANGED (scalar, outside the core).
// CORE256: call_opaque "(float)*(const _Float16 *)"
// The mf2 half-block setvl + the packed weight u8 load + the two q8 signed halves.
// CORE256: call_opaque "__riscv_vsetvl_e8mf2"
// CORE256: call_opaque "__riscv_vle8_v_u8mf2"
// CORE256: call_opaque "__riscv_vle8_v_i8mf2"
// CORE256: call_opaque "__riscv_vle8_v_i8mf2"
// The codebook nibble split + gather at mf2.
// CORE256: call_opaque "__riscv_vand_vx_u8mf2"
// CORE256: call_opaque "__riscv_vsrl_vx_u8mf2"
// CORE256: call_opaque "__riscv_vrgather_vv_i8mf2"
// CORE256: call_opaque "__riscv_vrgather_vv_i8mf2"
// The widened product one step narrower than the m1 form: i16m1 (VLEN128 emits i16m2).
// CORE256: call_opaque "__riscv_vwmul_vv_i16m1"
// CORE256: call_opaque "__riscv_vwmacc_vv_i16m1"
// The reduction destination + seed STAY m1: vwredsum_vs_i16m1_i32m1, NOT _i16m2_.
// CORE256: call_opaque "__riscv_vwredsum_vs_i16m1_i32m1"
// CORE256-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE256: call_opaque "__riscv_vmv_x_s_i32m1_i32"

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>), identical at both
// anchors (the VLEN flip changes the body bytes, not the ABI symbol).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_rvv_mxfp4_q8_0_block_dot
