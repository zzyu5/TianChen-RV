// FULL production-export CLOSURE for the iq4_xs (ggml IQ4_XS x Q8_K super-block
// CODEBOOK block-dot) front door -- the P2-c coverage payoff that closes the LAST
// of the four format-buckets: the SUPER-BLOCK-CODEBOOK bucket (the intersection of
// q4_K's super-block structure and iq4_nl's 16-entry codebook gather). The front
// door's OWN auto-constructed typed super-block SCALAR-accumulator loop body (fold_model
// "scalar_delta_grid", the iq4_xs CODEBOOK integer-core brick -- the flip retired the
// monolith op) now flows
// through the COMPLETE weft-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --weft-check-execution-plan-coherence) AND
// exports a real RISC-V target artifact through weft-translate
// --weft-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block-codebook is
// MECHANICAL, NOT bespoke. iq4_xs takes the EXISTING super-block monolithic route
// family (shared with q4_K, the 'rvv-ggml-super-block-block-dot-monolithic-emitc-
// route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The codebook is an OP attr consumed by the emitter,
// NOT a route-family concern: the emission plan (buildMonolithicBlockDotEmission
// Plan) and the target-export candidate validator key ONLY off op name -> route
// family + the kind/scale_model attrs + the ordered ABI roles; neither reads the
// codebook. So the shared SuperBlock family + iq4_nl's codebook-attr stamping
// COMPOSE cleanly: COVERAGE = one table row (RVVMonolithicBlockDotFamily.h,
// SuperBlock + the 4-role ggml vec_dot ABI n/s/vx/vy) + one front door
// (RVVIQ4XSBlockDotSourceFrontDoor.cpp), NOT any new mechanism. q4_K/q4_0/q8_0/
// iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq4_xs's front-door-constructed op is
// left attr-less (no shape knob); iq4_xs is NOT in any schedule autotuner, and the
// codebook gather pins m1 in the emitter (the broadcast codebook register's VLMAX
// must be >= 16 to index all 16 table entries), so the op lowers at its m1
// integer-core anchor -- there is NO VLEN128-vs-VLEN256 byte-flip for iq4_xs.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the typed super-block codebook-core loop
// block-dot body, the weft-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --weft-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the iq4_nl CODEBOOK gather (the 4-bit nibble
// INDEXES the 16-entry non-linear int8 table via vrgather -- NOT a linear nibble-8
// decode), wrapped in the super-block scale machinery -- pinned so a regression
// into an arithmetic decode or a wrong scale domain is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_iq4_xs_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_iq4_xs_q8_K_kernel"} {
  func.func @source_iq4_xs_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_iq4_xs_q8_K_kernel
// iq4_xs FLIP (C_construct 23->24, the FIRST super-block CODEBOOK member): the front door
// now constructs the typed super-block SCALAR-accumulator loop body (fold_model
// "scalar_delta_grid", stride 136) with the iq4_xs CODEBOOK integer-core brick (iq4_nl's
// 16-entry vrgather codebook gather + the q4_K-style signed 6-bit scale bit-dance), NOT the
// retired monolith op. The export still resolves through the SAME super-block monolithic
// route family (kind/ABI/facts) by the selector + weight_block_stride 136, so the
// emission-plan metadata is byte-unchanged.
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq4_xs kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq4_xs_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq4_xs_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC codebook integer core ======================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot
// The codebook emitted as a structured static const int8_t[16] decl (the SAME
// non-linear nibble->int8 lookup table kvalues_iq4nl[16] iq4_nl uses).
// CORE: verbatim "static const int8_t weft_iq4_xs_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The codebook table broadcast-loaded ONCE (above the super-block loop), i8m1.
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The two scalar block-scale reads -> d4d8 = fp16(x.d) * fp32(y.d).
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The SIGNED 6-bit scale extraction: scalar emitc bitwise shift/and/or (no string).
// CORE: bitwise_right_shift
// CORE: bitwise_and
// CORE: bitwise_left_shift
// CORE: bitwise_or
// d1 = d4d8 * (float)(ls - 32): a SEPARATE sub + cast + mul (NOT fused with sumi).
// CORE: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// The CODEBOOK mechanism: split the nibble into the two UNSIGNED index lanes, then
// GATHER each through the broadcast table (NOT an arithmetic nibble-8 decode).
// CORE: call_opaque "__riscv_vand_vx_u8m1"
// CORE: call_opaque "__riscv_vsrl_vx_u8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// CORE: call_opaque "__riscv_vrgather_vv_i8m1"
// The shared asymmetric widening product + seed-0 vwredsum + scalar extract.
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwmacc_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The FLOAT-domain fold: ONE expression d1 * (float)sumi then sumf + that (iq4_xs
// is SYMMETRIC -- NO min term, NO integer-domain aux32).
// CORE: expression : !emitc.opaque<"float">
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot
