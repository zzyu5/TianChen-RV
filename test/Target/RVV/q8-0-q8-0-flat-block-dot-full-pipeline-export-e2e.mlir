// FULL production-export CLOSURE for the q8_0 (ggml Q8_0 x Q8_0 FLAT block-dot)
// front door -- the P2-c coverage payoff that brings the FAMILY-A sibling of the
// q4_0 flat block-dot through the SAME generalized monolithic wiring q4_K/q4_0/
// iq4_nl proved. The front door's OWN auto-constructed monolithic flat block-dot
// body now flows through the COMPLETE tcrv-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --tcrv-check-execution-plan-coherence) AND
// exports a real RISC-V target artifact through tcrv-translate
// --tcrv-export-target-artifact.
//
// COVERAGE = one more table row (RVVMonolithicBlockDotFamily.h) + one more front
// door (RVVQ80BlockDotSourceFrontDoor.cpp), NOT any new mechanism. The recognition,
// the SUPER-BLOCK vs FLAT route-family split, and the per-op ABI table are shared;
// q8_0 is a FLAT op, so it takes the flat route id
// 'rvv-ggml-flat-block-dot-monolithic-emitc-route-family' (NOT q4_K's super-block
// route id) and carries the FULL 8-role strided ggml vec_dot ABI (n, s, bs, vx, bx,
// vy, by, nrc) -- byte-identical shape to q4_0's, because q8_0's vec_dot prototype
// is byte-identical to q4_0's. q4_K/q4_0/iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q8_0's front-door-constructed body is
// the typed flat block-dot loop body, which pins the m2 integer-core anchor (the
// ggml-matching one-vwredsum-per-block anchor) -- there is NO VLEN128-vs-VLEN256
// byte-flip for q8_0.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic flat block-dot body,
// the tcrv-source-artifact-front-door-pipeline materializes the emission plan AND
// passes --tcrv-check-execution-plan-coherence (the flat monolithic route id is
// now a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the flat monolithic emission plan exports a real
// RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the plain signed i8 x i8 widening product (NO
// nibble decode / NO offset-binary bias / NO half-block split), the q8_0 sibling
// distinction -- pinned so a regression into the q4_0 nibble core is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_q8_0_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel"} {
  func.func @source_q8_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the FLAT monolithic route id + object kind.
// The q8_0 front door's body is the typed flat block-dot LOOP body op (M-FLAT);
// it exports through the SAME shared q8_0 Flat monolithic plan (identical route
// id / kind / 8-role ABI) as the compound q8_0 block-dot op it replaced.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel
// PLAN: tcrv_rvv.typed_flat_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The flat block-dot carries the flat (not super-block) op-derived metadata keys
// (rendered inside artifact_metadata, ahead of lowering_pipeline).
// PLAN-SAME: rvv_ggml_flat_block_dot_kind
// PLAN-SAME: ggml_q8_0_q8_0_block_dot
// The honest FLAT monolithic-body route id (NOT q4_K's super-block route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-flat-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q8_0_q8_0_block_dot
// The flat block-dot honestly carries NO decomposed-route slice config metadata,
// and never claims the super-block route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC integer core (plain int8) ==================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot
// The two scalar fp16->fp32 block-scale reads.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: call_opaque "(float)*(const _Float16 *)"
// The plain i8m2 x i8m2 widening product into i16m4 (the m2 default anchor; NO
// vxor/vsll/vsra nibble decode, NO vwmacc high-half MAC).
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vle8_v_i8m2"
// CORE: call_opaque "__riscv_vwmul_vv_i16m4"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CORE-NOT: call_opaque "__riscv_vxor_vx_i8
// CORE-NOT: call_opaque "__riscv_vwmacc_vv_i16
// The q8_0 PINNED SeparatedLeftAssoc fold [testing/flat-block-dot-fp-fold-oracle.md
// §1]: (float)sumi cast, then t=(float)sumi*d_x, then t=t*d_y, then sumf=sumf+t --
// SEPARATE cast/mul/mul/add statements (NO d_x*d_y premultiply, NO fused
// emitc.expression, so clang cannot contract into fmaf). The assign takes the ADD
// result directly (locks separation).
// CORE: %[[FC:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CORE: %[[FT:.*]] = mul %[[FC]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: %[[FT2:.*]] = mul %[[FT]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: %[[FACC:.*]] = add %{{.*}}, %[[FT2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: assign %[[FACC]] : !emitc.opaque<"float"> to

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot
