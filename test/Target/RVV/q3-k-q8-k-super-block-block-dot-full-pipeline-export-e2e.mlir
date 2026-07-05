// FULL production-export CLOSURE for the q3_K (ggml Q3_K x Q8_K super-block
// block-dot) front door -- the P2-e super-block-PLAIN K-quant rung (q3_K is the
// 3-bit modern K-quant, the LAST common K-quant). The front door's OWN
// auto-constructed monolithic super-block block-dot body now flows through the
// COMPLETE tcrv-source-artifact-front-door-pipeline (materialize-emission-plans
// PLUS --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target
// artifact through tcrv-translate --tcrv-export-target-artifact. This takes q3_K
// from "emits through the CORE lowering" to "fully production-reachable":
// coherence recognizes the SUPER-BLOCK monolithic route id and the target
// artifact exports byte-identical to the CORE emit.
//
// THE BUCKET VERDICT (why this exemplar is COVERAGE, not a new mechanism): q3_K
// is a genuine super-block (QK_K == 256), so it takes the EXISTING super-block
// monolithic route family (shared with q4_K/iq4_xs, the 'rvv-ggml-super-block-
// block-dot-monolithic-emitc-route-family' route id + the super-block op-derived
// metadata keys), NOT a new route-id / family variant. q3_K's block-format delta
// -- the 3-bit weight = 2-bit qs plane PLUS the SUBTRACTIVE hmask high-bit plane
// (`a = (low2 | (hbit<<2)) - 4` -> SIGNED [-4,3]), the q3_K-OWN SIGNED 6-bit
// scale dance (`scale_j = scales[j] - 32`), and q6_K's NO-min deferred symmetric
// fp32 fold (q3_K is SYMMETRIC -- NO min term, NO dmin, NO bsums) -- is op
// STRUCTURE the emitter consumes, NOT a route-family or ABI concern. The emission
// plan + the target-export candidate validator key ONLY off op name -> route
// family + kind/scale_model + the ordered 4-role ggml vec_dot ABI (n, s, vx, vy),
// none of which q3_K perturbs. So COVERAGE = one table row (RVVMonolithicBlockDot
// Family.h, SuperBlock + 4-role ABI) + one front door (RVVQ3KBlockDotSource
// FrontDoor.cpp), NOT any new mechanism. q4_K/iq4_xs/q4_0/q8_0/iq4_nl stay
// byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same
// tryConvertModuleWithRegisteredBackend / convertRVVModuleToEmitC lowering the
// direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q3_K's front-door-constructed op is
// left attr-less (no shape knob): q3_K has NO VLEN128-vs-VLEN256 byte-flip and is
// NOT in any schedule autotuner, so the op lowers at the emitter's DEFAULT mf2
// integer-core anchor (the op's integer_core_lmul "m1" knob is a DORMANT,
// emitter-sealed knob, never stamped here).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block block-dot
// body, the tcrv-source-artifact-front-door-pipeline materializes the emission
// plan AND passes --tcrv-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q3-k-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q3-k-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q3-k-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q3-k-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is q3_K's 3-bit decode: the 2-bit qs plane OR'd
// with the SUBTRACTIVE hmask high-bit plane (then `-4` -> signed), the q3_K-OWN
// SIGNED 6-bit scale dance, and q6_K's NO-min deferred symmetric fp32 fold --
// pinned so a regression into a wrong (non-subtractive) hmask, an unsigned scale,
// or a fused/min fold is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_q3_K_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q3_K_q8_K_kernel"} {
  func.func @source_q3_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// The front door now auto-constructs the CONSTRUCTED typed SINGLE-accumulator loop
// body (tcrv_rvv.typed_super_block_block_dot_loop_body, fold_model
// "scales_times_sumi" -- the q3_K aux32 integer core + the reused no-min positive
// fold), NOT an opaque tcrv_rvv.q3_k_q8_k_block_dot op (retired same action as the
// flip). It resolves to q3_K's OWN export entry (kind / route id / target) by
// fold_model + weight_block_stride 110, so the export plan below is unchanged.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q3_K_q8_K_kernel
// PLAN: tcrv_rvv.typed_super_block_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the q3_K kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_q3_k_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q3_K_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC q3_K 3-bit integer core =====================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot
// The int8_t aux8[256] + uint32_t utmp[4] super-block scratch, and the carried
// 8-lane fp32 sums vector (the q6_K no-min deferred fold; NO scalar sumf).
// CORE: !emitc.array<256x!emitc.opaque<"int8_t">>
// CORE: !emitc.array<4x!emitc.opaque<"uint32_t">>
// CORE: call_opaque "__riscv_vfmv_v_f_f32m2"
// The 2-bit + SUBTRACTIVE-hmask unpack: u8m2 qs load, vsrl/vand the low 2 bits,
// vsrl/vand/vsll the hmask high bit, vor them, u8->i8 reinterpret, vsub 4 (the
// SUBTRACTIVE bias -> signed [-4,3]), vse8.
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vand_vx_u8m2"
// CORE: call_opaque "__riscv_vsll_vx_u8m2"
// CORE: call_opaque "__riscv_vor_vv_u8m2"
// CORE: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CORE: call_opaque "__riscv_vsub_vx_i8m2"
// CORE: call_opaque "__riscv_vse8_v_i8m2"
// The q3_K SIGNED 6-bit scale dance (STRUCTURED scalar emitc; NO raw()).
// CORE: bitwise_and
// CORE: bitwise_or
// CORE: bitwise_left_shift
// CORE: bitwise_right_shift
// The per-sub-block widening dot into the 8-lane aux32 at the DEFAULT mf2 anchor:
// the SIGNED scale `(int)sc[js] - 32` (a sub on the loaded int8), then i8mf2
// loads -> vwmul i16m1 -> vwmacc.vx i32m2.
// CORE: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: call_opaque "__riscv_vsetvl_e8mf2"
// CORE: call_opaque "__riscv_vle8_v_i8mf2"
// CORE: call_opaque "__riscv_vwmul_vv_i16m1"
// CORE: call_opaque "__riscv_vwmacc_vx_i32m2"
// The NO-min DEFERRED symmetric fp32 fold (q6_K's): the fp16 read seam ONCE (NO
// dmin second read), vfcvt i32->f32, a SEPARATE vfmul then a SEPARATE vfadd
// (NEVER a fused vfmacc/vfmadd), then the post-loop vse32 + SEQUENTIAL horizontal
// sum (NOT a vfredusum).
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CORE: call_opaque "__riscv_vfmul_vf_f32m2"
// CORE: call_opaque "__riscv_vfadd_vv_f32m2"
// CORE-NOT: call_opaque "__riscv_vfmacc
// CORE-NOT: call_opaque "__riscv_vfmadd
// CORE-NOT: call_opaque "__riscv_vfredusum
// CORE: call_opaque "__riscv_vse32_v_f32m2"
// CORE: return
// The residual operator-identity source func lowers to NOTHING: exactly ONE kernel.
// CORE-NOT: emitc.func @tcrv_emitc_source_q3_K_q8_K_block_dot

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot
