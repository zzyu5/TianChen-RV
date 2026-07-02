// FULL production-export CLOSURE for the q2_K (ggml Q2_K x Q8_K super-block
// block-dot) front door -- one of the four super-block-PLAIN K-quants (q2_K/q3_K/
// q5_K/q6_K, P2-e) that take the SAME super-block monolithic route family as q4_K.
// The front door's OWN auto-constructed monolithic super-block block-dot body now
// flows through the COMPLETE tcrv-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --tcrv-check-execution-plan-coherence) AND
// exports a real RISC-V target artifact through tcrv-translate
// --tcrv-export-target-artifact.
//
// WHY q2_K is MECHANICAL, NOT bespoke: q2_K is a genuine super-block (QK_K == 256)
// so it reuses the EXISTING super-block monolithic route family (the shared
// 'rvv-ggml-super-block-block-dot-monolithic-emitc-route-family' route id + the
// super-block op-derived metadata keys), NOT a new route-id / family variant. The
// block-format delta -- 2-bit weights (`(qs >> shift) & 3` over shifts {0,2,4,6}),
// the SIMPLE 4-bit nibble scale/min of the direct scales[16] bytes (NO 6-bit
// utmp/kmask bit-dance), and the SCALAR fp32 fold (`sumf += dall*isum - dmin*summs`,
// NO 8-lane deferred sums vector, NO post-loop horizontal sum) -- is op STRUCTURE
// the emitter consumes, NOT a route-family or ABI concern. Its 4-role ggml vec_dot
// ABI (n, s, vx, vy) matches q4_K's. So the shared SuperBlock family + q2_K's
// block-format constants COMPOSE cleanly: COVERAGE = one table row
// (RVVMonolithicBlockDotFamily.h) + one front door (RVVQ2KBlockDotSourceFrontDoor
// .cpp), NOT any new mechanism. q4_K/iq4_xs/q4_0/q8_0/iq4_nl stay byte-exact on
// their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q2_K's front-door-constructed op is
// left attr-less (q2_K carries NO shape knob at all -- no integer_core_lmul, unlike
// q4_K/q6_K); q2_K is NOT in any schedule autotuner, so the op lowers at the q2_K
// emitter's DEFAULT integer-core anchor -- there is NO VLEN128-vs-VLEN256 byte-flip
// for q2_K.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block block-dot
// body, the tcrv-source-artifact-front-door-pipeline materializes the emission
// plan AND passes --tcrv-check-execution-plan-coherence (the super-block monolithic
// route id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the q2_K 2-bit unpack (`(qs >> shift) & 3`) +
// SIMPLE 4-bit nibble scale/min + the SCALAR fp32 fold -- pinned so a regression
// into a 6-bit bit-dance, an 8-lane deferred sums vector, or a wrong scale domain
// is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_q2_K_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q2_K_q8_K_kernel"} {
  func.func @source_q2_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_q2_K_q8_K_kernel
// PLAN: tcrv_rvv.q2_k_q8_k_block_dot
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the q2_K kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_q2_k_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q2_K_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC q2_K 2-bit integer core ====================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot
// The int8_t aux8[256] element-ordered scratch.
// CORE: !emitc.array<256x!emitc.opaque<"int8_t">>
// The scalar sumf, zeroed ONCE outside the super-block loop (NO 8-lane sums vector
// -- q2_K's positive term is the scalar isum).
// CORE: !emitc.lvalue<!emitc.opaque<"float">>
// The 2-bit weight unpack: u8m2 load of the qs chunk, then vand/vsrl over the shifts
// {0,2,4,6}, u8->i8 reinterpret, vse8.
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vand_vx_u8m2"
// CORE: call_opaque "__riscv_vsrl_vx_u8m2"
// CORE: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CORE: call_opaque "__riscv_vse8_v_i8m2"
// The SIMPLE 4-bit nibble scale/min of the direct scales[16] bytes (NO 6-bit
// utmp/kmask bit-dance -- so NO bitwise_or here).
// CORE-NOT: bitwise_or
// CORE: bitwise_and
// CORE: bitwise_right_shift
// The per-sub-block widening dot into the scalar isum: i8m1 loads -> vwmul ->
// vwredsum -> vmv_x_s.
// CORE: call_opaque "__riscv_vsetvl_e8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SCALAR fp32 fold: dall/dmin via the fp16 read seam, then
// `sumf += dall*isum - dmin*summs` as ONE emitc.expression (two muls + a sub + an
// add). NO 8-lane vfcvt/vfmul/vfadd, NO post-loop vse32 horizontal sum.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: expression
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CORE-NOT: call_opaque "__riscv_vfmacc
// CORE-NOT: call_opaque "__riscv_vfmadd
// CORE-NOT: call_opaque "__riscv_vse32_v_f32m2"
// CORE-NOT: call_opaque "__riscv_vfredusum

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot
