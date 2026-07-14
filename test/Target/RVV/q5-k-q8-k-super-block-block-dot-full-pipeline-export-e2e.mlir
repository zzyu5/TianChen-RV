// FULL production-export CLOSURE for the q5_K (ggml Q5_K x Q8_K super-block
// block-dot) front door -- the P2-e coverage rung: q5_K is q4_K PLUS a 5th (high)
// weight bit (the qh plane). The front door's OWN auto-constructed monolithic
// super-block block-dot body now flows through the COMPLETE
// weft-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --weft-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through weft-translate --weft-export-target-artifact. This takes q5_K from
// "emits through the emission-plans stage" to "fully production-reachable":
// coherence recognizes the monolithic route id and the target artifact exports
// byte-identical to the CORE emit.
//
// THE BUCKET VERDICT (why this exemplar matters): the super-block-PLAIN K-quant is
// MECHANICAL, NOT bespoke. q5_K takes the EXISTING super-block monolithic route
// family (shared with q4_K/q2_K/q3_K/q6_K/iq4_xs, the 'rvv-ggml-super-block-block-
// dot-monolithic-emitc-route-family' route id + the super-block op-derived metadata
// keys), NOT a new route-id / family variant. The qh high-bit plane is op structure
// consumed by the emitter, NOT a route-family concern: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator
// key ONLY off op name -> route family + the kind/scale_model attrs + the ordered
// ABI roles; none reads the qh plane. So the shared SuperBlock family + q5_K's
// qh-plane-attr stamping COMPOSE cleanly: COVERAGE = one table row
// (RVVMonolithicBlockDotFamily.h, SuperBlock + the 4-role ggml vec_dot ABI
// n/s/vx/vy) + one front door (RVVQ5KBlockDotSourceFrontDoor.cpp), NOT any new
// mechanism. q4_K/q4_0/q8_0/iq4_nl/iq4_xs stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. q5_K's front-door-constructed op is
// left attr-less (no shape knob); q5_K is NOT in any schedule autotuner and carries
// no integer_core_lmul knob at all, so the op lowers at the emitter's DEFAULT mf2
// integer-core anchor -- there is NO VLEN128-vs-VLEN256 byte-flip for q5_K.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the monolithic super-block block-dot
// body, the weft-source-artifact-front-door-pipeline materializes the emission plan
// AND passes --weft-check-execution-plan-coherence (the super-block monolithic route
// id is a registered target-artifact export route).
// RUN: weft-opt %s --weft-rvv-materialize-q5-k-q8-k-block-dot-source-front-door --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-materialize-q5-k-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-materialize-q5-k-q8-k-block-dot-source-front-door --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q5-k-q8-k-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is q4_K's K4b super-block machinery (the plain 4-bit
// nibble unpack into aux8[256], the 6-bit scale/min bit-dance via utmp, the deferred
// fp32 fold + MIN term) PLUS the ONE q5_K-specific piece -- the qh 5th-bit plane
// injection -- pinned so a regression into q4_K's plain unpack (missing the qh
// injection) or a wrong fold order is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {weft_rvv.source_front_door = "ggml_q5_K_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q5_K_q8_K_kernel"} {
  func.func @source_q5_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported SUPER-BLOCK
// emission-plan diagnostic naming the SUPER-BLOCK route id + object kind. After the
// q5_K FLIP the front door constructs the TYPED super-block dual-accumulator loop
// body (the 5 shared q4_K/q5_K bricks + BRICK 1's qh 5th-bit offset), NOT an opaque
// weft_rvv.q5_k_q8_k_block_dot op; the export entry is resolved by the loop op's
// weight_block_stride (176 -> the q5_K entry) so kind/ABI/route stay q5_K.
// PLAN: weft.exec.kernel @ggml_vec_dot_q5_K_q8_K_kernel
// PLAN: weft_rvv.typed_super_block_block_dot_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the q5_K kind -- the SAME super-block route family q4_K uses.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_q5_k_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_q5_K_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC qh-plane integer core ======================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot
// The super-block scratch: the int8_t aux8[256] + uint32_t utmp[4] + float sums8[8].
// CORE: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// CORE: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// CORE: %[[SUMS8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"float">>
// The shared K4b integer core: the plain 4-bit nibble unpack (first u8m2 load).
// CORE: call_opaque "__riscv_vsetvl_e8m2"
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// The q5_K-SPECIFIC piece: the qh 5th-bit plane loaded (a SECOND u8m2 load from a
// FIXED per-super-block offset) and injected via the SHARED [QH-MASK] native-mask
// helper -- the SINGLE qh bit isolated IN PLACE (vand 1<<h), lifted to a per-lane
// bool (vmsne==0, SET lanes), and the +16 FUSED into ONE vadd_vx_u8m2_mu on the SET
// lanes -- in the u8 domain BEFORE the u8->i8 reinterpret (lifting q4 in [0,15] to q5
// in [0,31]). This is the ONLY q5_K-vs-q4_K node difference; a regression into q4_K's
// plain unpack drops the vmsne/vadd_mu pair. Byte-exact to the RETIRED vsll<<4|vadd_vv.
// CORE: call_opaque "__riscv_vle8_v_u8m2"
// CORE: call_opaque "__riscv_vand_vx_u8m2"
// CORE: call_opaque "__riscv_vand_vx_u8m2"
// CORE: call_opaque "__riscv_vmsne_vx_u8m2_b4"
// CORE: call_opaque "__riscv_vadd_vx_u8m2_mu"
// CORE: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CORE: call_opaque "__riscv_vse8_v_i8m2"
// The OLD per-lane qh contribution chain (vsll<<4 then a vector-vector vadd of the
// 0/16 contribution) is fully RETIRED by the native-mask helper.
// CORE-NOT: call_opaque "__riscv_vadd_vv_u8m2"
// The 6-bit scale/min bit-dance (scalar emitc bitwise ops; NOT the K4a 16-byte store).
// CORE: bitwise_right_shift
// CORE: bitwise_and
// CORE: bitwise_or
// CORE-NOT: call_opaque "__riscv_vse8_v_u8m1"
// The per-sub-block uint6-scaled i32 accumulation into the carried 8-lane aux32.
// CORE: call_opaque "__riscv_vwmul_vv_i16m1"
// CORE: call_opaque "__riscv_vwmacc_vx_i32m2"
// The DEFERRED two-level fp32 positive fold: SEPARATE vfmul then SEPARATE vfadd
// (NEVER a fused vfmacc), preceded by the fp16 scale read.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CORE: call_opaque "__riscv_vfmul_vf_f32m2"
// CORE: call_opaque "__riscv_vfadd_vv_f32m2"
// The q5_K MIN term sumf -= dmin*sumi as ONE emitc.expression (a mul + a sub).
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: expression
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CORE-NOT: call_opaque "__riscv_vfmacc
// CORE-NOT: call_opaque "__riscv_vfmadd
// The SEQUENTIAL horizontal sum: vse32 the lanes into sums8, then scalar adds (NOT a
// vfredusum), then the *s store.
// CORE: call_opaque "__riscv_vse32_v_f32m2"
// CORE-NOT: call_opaque "__riscv_vfredusum
// CORE-NOT: call_opaque "__riscv_vfredosum
// CORE: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot
