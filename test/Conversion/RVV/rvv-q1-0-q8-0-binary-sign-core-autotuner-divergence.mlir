// The q1_0 Win-A proof starts at the public source front door. The front door
// constructs the sole live typed authority:
//
//   source identity -> typed_flat_block_dot_loop_body
//                   -> q1_0_q8_0_binary_sign_core
//                   -> typed-loop emitter.
//
// No retired whole-kernel q1_0 op is parsed or emitted here. The SAME attr-less
// binary-sign core is then scheduled by the unified TunableSchedule interface.
// Its 32-lane sub-block requires m2 at VLEN128, while m1 is legal and lighter at
// VLEN256.
//
// Decision-level proof: the source front door constructs exact P; after generic
// selection, the RVV owner constructs different final plans from only c_o.
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Emission-level non-NULL proof: the typed-loop emitter consumes those stamped
// choices and produces byte-different intrinsic shapes.
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door=march=rv64gcv --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: weft-opt %s --weft-rvv-materialize-q1-0-q8-0-block-dot-source-front-door=march=rv64gcv_zvl256b --weft-execution-planning-pipeline --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

module attributes {
  weft_rvv.source_front_door = "ggml_q1_0_q8_0_block_dot_source",
  weft_rvv.source_kernel = "ggml_vec_dot_q1_0_q8_0_kernel"
} {
  func.func @source_q1_0_q8_0_block_dot(
      %s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ================= STAMPED TYPED AUTHORITY ==================
// At VLEN128, m2 is the smallest legal anchor.
// STAMP-VLEN128: weft_rvv.typed_flat_block_dot_loop_body
// STAMP-VLEN128: weft_rvv.q1_0_q8_0_binary_sign_core
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-NOT: weft_rvv.q1_0_schedule.
// STAMP-VLEN128-NOT: weft_rvv.q1_0_q8_0_block_dot
//
// At VLEN256, the SAME constructed core flips to m1.
// STAMP-VLEN256: weft_rvv.typed_flat_block_dot_loop_body
// STAMP-VLEN256: weft_rvv.q1_0_q8_0_binary_sign_core
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64
// STAMP-VLEN256-NOT: weft_rvv.q1_0_q8_0_block_dot

// ===================== VLEN128 — m2 typed emission ====================
// VLEN128: emitc.func @weft_emitc_ggml_vec_dot_q1_0_q8_0_kernel_rvv_q1_0_q8_0_block_dot(
// VLEN128: call_opaque "__riscv_vsetvl_e8m2"
// VLEN128: call_opaque "__riscv_vlm_v_b4"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vneg_v_i8m2"
// VLEN128: call_opaque "__riscv_vmerge_vvm_i8m2"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i8m2_i16m1"
// VLEN128-NOT: call_opaque "__riscv_vsetvl_e8m1"
// VLEN128-NOT: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// VLEN128: return

// ===================== VLEN256 — m1 typed emission ====================
// VLEN256: emitc.func @weft_emitc_ggml_vec_dot_q1_0_q8_0_kernel_rvv_q1_0_q8_0_block_dot(
// VLEN256: call_opaque "__riscv_vsetvl_e8m1"
// VLEN256: call_opaque "__riscv_vlm_v_b8"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vneg_v_i8m1"
// VLEN256: call_opaque "__riscv_vmerge_vvm_i8m1"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// VLEN256-NOT: call_opaque "__riscv_vlm_v_b4"
// VLEN256-NOT: call_opaque "__riscv_vwredsum_vs_i8m2_i16m1"
// VLEN256: return
