// FULL production-export CLOSURE for the q8_0 FLAT block-dot with the P2c
// fold_structure = "deferred-ordered" schedule -- the DEFERRED sibling of
// q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir. The shipped e2e file
// exports the PER-BLOCK m2 kernel the front door auto-constructs; this file
// exports the SAME kernel symbol but with the deferred batched-vector cross-block
// fold (multi_block_factor = 4, strip_elision = elided, integer_core_lmul = m2,
// fold_structure = deferred-ordered).
//
// WHY the body is spelled out here (not a source stub): the q8_0 source front
// door (createTypedFlatBlockDotLoopChain) hardcodes per-block + mbf=1 + no
// fold_structure, so it CANNOT construct the deferred body. The deferred config
// (q8_0 / m2 / elided / mbf=4 / deferred-ordered) is already legal + green in
// the emit gates and the verifier (see
// rvv-to-emitc-q8-0-q8-0-typed-flat-block-dot-loop-body-deferred.mlir), it is
// only unreachable from the front door. So this file carries the POST-front-door
// constructed module (kernel + rvv variant + scalar fallback + dispatch) VERBATIM
// as the front door emits it, plus the two deferred knobs on the typed body. The
// module carries NO weft_rvv.source_front_door attribute, so re-running the front
// door pass over it is a no-op -- the on-device harness (run_ondevice_verify.sh)
// can drive it unchanged (FRONT_DOOR_PASS no-ops, then materialize-emission-plans
// + weft-translate export the deferred .o).
//
// BYTE-EXACT vs the pinned §1 oracle is by construction (the deferred vfredosum.vs
// is seed-first / lane-ascending = the serial left fold); on-device confirmation
// (rvv board, VLEN128) recorded it 256/256 ULP=0.
//
// OBJECT-EXPORT NOTE (packaging-march gap, NOT wired here): unlike the per-block
// sibling, the deferred PHASE-A scale gather emits a VECTOR fp16 strided load +
// widen (__riscv_vlse16_v_f16mf2 -> __riscv_vfwcvt_f_f_v_f32m1), which needs the
// zvfhmin extension. The in-tree RVV object-packaging clang hardcodes
// -march=rv64gcv (RVVTargetSupportBundle.cpp compileRVVGeneratedSourceToObject),
// which lacks zvfhmin, so `weft-translate --weft-export-target-artifact` on this
// module CURRENTLY fails at the clang step ("requires the 'zvfh or zvfhmin'
// extension"). The rvv board's ISA carries zvfh/zvfhmin, so the kernel runs
// natively once packaged with a zvfhmin march. The early-signal .o was produced
// off-tree with clang -march=rv64gcv_zvfhmin. Wiring the packaging march to the
// kernel's capability requirements (so the deferred body exports through the
// unmodified pipeline) is the follow-up. The direct object-export RUN lines are
// therefore left commented until that lands:
//   weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
//   llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
//   llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

// The CORE EmitC carries the DEFERRED fold: the batched vfredosum.vs ordered
// reduction (NOT the per-block scalar fmaf round-trip), and NO FMA-family op.
// This runs in-tree (lower-to-emitc does not invoke the packaging clang).
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=CORE --implicit-check-not=vfredusum --implicit-check-not=vfmacc --implicit-check-not=vwmacc

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_q8_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %3 = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %4 = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %5 = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %6 = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %7 = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %8 = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %9 = weft_rvv.setvl %0 {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %9 attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q8_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %3, %5, %1, %0 attributes {activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", fold_structure = "deferred-ordered", integer_core_lmul = "m2", kind = "typed_flat_block_dot_loop_body", multi_block_factor = 4 : i64, numerics_tier = "strict", qk = 32 : i64, strip_elision = "elided", weight_block_stride = 34 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %10 = weft_rvv.block_fp16_scale_product %3, %5 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %11 = weft_rvv.load %3, %9 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %12 = weft_rvv.load %5, %9 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %13 = weft_rvv.widening_product %11, %12, %9 {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
          %14 = weft_rvv.standalone_reduce %13, %8, %9 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          %15 = weft_rvv.typed_vector_lane0_to_scalar_extract %14, %9 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
          %16 = weft_rvv.block_computed_scale_dequant %15, %10 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %17 = weft_rvv.cross_block_f32_accumulate %arg1, %16 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %17 : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
    weft.exec.variant @rvv_q8_0_q8_0_block_dot_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_q8_0_q8_0_block_dot {origin = "rvv-plugin", policy = "rvv-q8-0-q8-0-block-dot-source-front-door-case"}
      weft.exec.fallback @rvv_q8_0_q8_0_block_dot_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
  func.func @source_q8_0_q8_0_block_dot(%arg0: memref<?xf32>, %arg1: index, %arg2: memref<?xi8>, %arg3: memref<?xi8>) {
    return
  }
}

// ===================== CORE EmitC deferred fold ==============================
// CORE: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// The deferred PHASE-A pack (vslide1down) + PHASE-B ORDERED reduction seeded by
// the running sumf -- the batched fold, NOT a per-block scalar fmaf round-trip.
// CORE: call_opaque "__riscv_vslide1down_vx_i32m1"
// CORE: call_opaque "__riscv_vlse16_v_f16mf2"
// CORE: call_opaque "__riscv_vfmv_v_f_f32m1"
// CORE: call_opaque "__riscv_vfmul_vv_f32m1"
// CORE: call_opaque "__riscv_vfmul_vv_f32m1"
// CORE: call_opaque "__riscv_vfredosum_vs_f32m1_f32m1"

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported symbol is the SAME kernel+variant handoff name as the per-block
// shipped file -- deferred is a fold-schedule swap, not a new kernel.
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot
