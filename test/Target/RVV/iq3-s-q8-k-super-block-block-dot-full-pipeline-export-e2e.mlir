// FULL production-export CLOSURE for the iq3_s (ggml IQ3_S x Q8_K super-block
// GRID-CODEBOOK block-dot) front door -- the coverage payoff that closes the LAST
// iq* super-block-codebook bucket (the literal block-dot zoo). iq3_s is a
// RE-COMPOSITION of three already-built GRID-codebook mechanisms wrapped in the
// q6_K-style super-block structure: the iq3 512-entry GRID-of-4 codebook (each entry
// packs FOUR int8 values, an indexed vle8(4) over grid_table + idx*4, NOT a
// vrgather), the qh 9th-bit plane (a SINGLE bit mask 256), and the EXPLICIT per-group
// signs read from a dedicated signs[QK_K/8] memory region. The front door's OWN
// auto-constructed typed super-block SCALAR-accumulator GRID loop body (fold_model
// "scalar_delta_grid", the iq3_s GRID-of-4 explicit-signs grid-core brick -- the flip
// retired the monolith op) now flows through the
// COMPLETE tcrv-source-artifact-front-door-pipeline (materialize-emission-plans PLUS
// --tcrv-check-execution-plan-coherence) AND exports a real RISC-V target artifact
// through tcrv-translate --tcrv-export-target-artifact.
//
// THE BUCKET VERDICT (why this exemplar matters): super-block GRID-codebook is
// MECHANICAL, NOT bespoke. iq3_s takes the EXISTING super-block monolithic route
// family (shared with q4_K / iq4_xs, the 'rvv-ggml-super-block-block-dot-monolithic-
// emitc-route-family' route id + the super-block op-derived metadata keys), NOT a new
// route-id / family variant. The grid is an OP attr consumed by the emitter, NOT a
// route-family concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and
// the target-export candidate validator key ONLY off op name -> route family + the
// kind/scale_model attrs + the ordered ABI roles; neither reads the grid. So the
// shared SuperBlock family + iq3_s's grid-attr stamping COMPOSE cleanly: COVERAGE =
// one table row (RVVMonolithicBlockDotFamily.h, SuperBlock + the 4-role ggml vec_dot
// ABI n/s/vx/vy) + one front door (RVVIQ3SBlockDotSourceFrontDoor.cpp), NOT any new
// mechanism. q4_K/iq4_xs/q4_0/q8_0/iq4_nl stay byte-exact on their own routes.
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same lowering
// the direct --tcrv-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- this is coverage/wiring maturity. iq3_s's front-door-constructed op is
// left attr-less (no shape knob); iq3_s is NOT in any schedule autotuner, so the op
// lowers at its m1 integer-core anchor -- there is NO VLEN128-vs-VLEN256 byte-flip
// for iq3_s.
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: tianchenrv-local-rvv-object-clang

// FULL pipeline: front door auto-constructs the typed super-block GRID grid-core loop
// block-dot body, the tcrv-source-artifact-front-door-pipeline materializes the
// emission plan AND passes --tcrv-check-execution-plan-coherence (the super-block
// monolithic route id is a registered target-artifact export route).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --tcrv-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the block-dot body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --tcrv-rvv-lower-to-emitc emit.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc > %t.core.mlir
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the super-block monolithic emission plan exports
// a real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC integer core is the iq3 GRID-of-4 codebook (the 9-bit index loads
// FOUR int8 grid values per index via an indexed vle8(4) over grid_table + idx*4 --
// NOT a vrgather, the 2048-byte table cannot broadcast into a vreg), the qh 9th-bit
// injection (mask 256), and the EXPLICIT sign plane read from memory -- pinned so a
// regression into a vrgather lookup, a wrong index assembly, or a wrong sign source
// is caught.
// RUN: FileCheck %s --check-prefix=CORE < %t.core.mlir

module attributes {tcrv_rvv.source_front_door = "ggml_iq3_s_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_iq3_s_q8_K_kernel"} {
  func.func @source_iq3_s_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the SUPER-BLOCK monolithic route id + object kind.
// PLAN: tcrv.exec.kernel @ggml_vec_dot_iq3_s_q8_K_kernel
// iq3_s FLIP (C_construct 22->23, EXPLICIT-SIGNS variant): the front door now constructs
// the typed super-block SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid",
// stride 110) with the iq3_s GRID-of-4 explicit-signs grid-core brick, NOT the retired
// monolith op. The export still resolves through the SAME super-block monolithic route
// family (kind/ABI/facts) by the selector + weight_block_stride 110, so the emission-plan
// metadata is byte-unchanged.
// PLAN: tcrv_rvv.typed_super_block_block_dot_loop_body
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The super-block block-dot carries the super-block (not flat) op-derived metadata
// keys, with the iq3_s kind -- the SAME super-block route family q4_K/iq4_xs use.
// PLAN-SAME: rvv_ggml_super_block_block_dot_kind
// PLAN-SAME: ggml_iq3_s_q8_k_block_dot
// The honest SUPER-BLOCK monolithic-body route id (NOT the flat route, NOT the
// decomposed generic-typed-body route) is the coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-super-block-block-dot-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_iq3_s_q8_K_block_dot
// The super-block block-dot honestly carries NO decomposed-route slice config
// metadata, and never claims the flat route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== CORE EmitC GRID-codebook integer core =================
// CORE: emitc.func @tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_rvv_iq3_s_q8_K_block_dot
// The GRID-of-4 codebook emitted as a structured static const uint32_t[512] decl
// (NOT vrgather: byte-viewed via (const int8_t *)), plus the inline {1<<j} kmask.
// CORE: verbatim "static const uint32_t tcrv_iq3s_grid[512] = {0x01010101U,
// CORE: verbatim "static const uint8_t tcrv_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CORE: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CORE: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The FULL 8-bit kmask broadcast-loaded ONCE (above the super-block loop), u8m1.
// CORE: call_opaque "__riscv_vle8_v_u8m1"
// The outer super-block loop.
// CORE: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CORE: call_opaque "(float)*(const _Float16 *)"
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CORE: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The explicit two-nibble scale split + the qh 9th-bit injection (mask 256, shift).
// CORE: bitwise_left_shift
// CORE: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CORE: bitwise_or
// The GRID-of-4 GROUP gather: the uint16_t tmp[2] idx*4 byte-offsets, vle16 index,
// then the HARDWARE __riscv_vluxei16_v_i32m1 over the i32 grid base, reinterpreted to
// i8m1 grid bytes, then the 8-lane q8 load (NOT a vrgather).
// CORE: "emitc.variable"() {{.*}} -> !emitc.array<2x!emitc.opaque<"uint16_t">>
// CORE: call_opaque "__riscv_vle16_v_u16mf2"
// CORE: call_opaque "__riscv_vluxei16_v_i32m1"
// CORE: call_opaque "__riscv_vreinterpret_v_i32m1_i8m1"
// CORE: call_opaque "__riscv_vle8_v_i8m1"
// The EXPLICIT SIGN plane: broadcast signs / vand kmask / vmsne / vneg / vmerge.
// CORE: call_opaque "__riscv_vmv_v_x_u8m1"
// CORE: call_opaque "__riscv_vand_vv_u8m1"
// CORE: call_opaque "__riscv_vmsne_vx_u8m1_b8"
// CORE: call_opaque "__riscv_vneg_v_i8m1"
// CORE: call_opaque "__riscv_vmerge_vvm_i8m1"
// The signed widening product + the chained vwredsum + scalar extract.
// CORE: call_opaque "__riscv_vwmul_vv_i16m2"
// CORE: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CORE: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// bsum += sumi*ls (integer domain).
// CORE: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fold: ONE expression d*(float)bsum then sumf + that.
// CORE: expression : !emitc.opaque<"float">
// CORE: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The *s store (NO trailing factor).
// CORE: subscript
// CORE: assign

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same
// name the CORE EmitC emit carries (tcrv_emitc_<kernel>_<variant>).
// SYMBOL: Name: tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_rvv_iq3_s_q8_K_block_dot
