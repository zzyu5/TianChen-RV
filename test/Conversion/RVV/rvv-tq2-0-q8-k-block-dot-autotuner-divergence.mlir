// The tq2_0 Win-A proof (PRESERVED across the tq2_0 flip): the COMPILER SELECTS the
// ggml TQ2_0 x Q8_K FUSED 2-bit ternary integer-core anchor, and the selection DIVERGES
// by capability from the SAME attr-less input -- via the UNIFIED schedule autotuner (the
// SAME walk-all pass that auto-discovers every TunableScheduleOpInterface op, NO
// per-tq2_0 pass). The monolith op was RETIRED at the flip; the Win-A gearbox moved
// verbatim onto the CONSTRUCTED weft_rvv.tq2_0_q8_k_ternary_core brick (SAME kernel key
// "tq2_0"), so the autotuner stamps the SAME m2->m1 selection onto the brick with NO
// registry change. This test now drives the CONSTRUCTED typed super-block
// SCALAR-accumulator TERNARY loop body (fold_model "scalar_delta_grid"), not the retired
// monolith op.
//
// The ternary-core brick below carries NO integer_core_lmul knob -- the compiler must
// compute it. tq2_0's fused ternary dot is ALWAYS one 32-lane plane body (load the 32-byte
// qs chunk once, 4 2-bit planes each vwmacc 32 ternary*q8 lanes into a wide i16
// accumulator zeroed ONCE, ONE vwredsum for the whole super-block). The single
// vsetvl_e8<anchor>(32) cover is
// correct ONLY at the whole-LMUL anchor whose i8 strip VLMAX spans the 32-element
// plane at the derived minimum VLEN. WHICH anchor that is MOVES with VLEN:
//
//   * at VLEN 128: only m2 spans it (e8m1 VLMAX 16 < 32) -> integer_core_lmul "m2"
//     (wide accumulator i16m4).
//   * at VLEN 256: m1 also reaches 32; m1 TIES m2 on the capability-blind cost and
//     the lighter footprint breaks the tie -> integer_core_lmul "m1" (i16m2 acc).
//
// One capability fact (the REAL VLEN bits) -> the anchor FLIPS m2->m1. This is the
// compiler SELECTING the shape from a capability fact, not a hand-set attr.

// First, the DECISION-LEVEL proof: the unified autotuner stamps DIFFERENT anchors
// onto the SAME attr-less ternary-core brick purely by the VLEN capability fact (no
// lowering).
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Then the EMISSION-LEVEL non-NULL proof: VLEN256 emits a BYTE-DIFFERENT kernel
// from VLEN128 (vsetvl_e8m1 / vle8_v_u8m1 / vwmacc_vv_i16m2 / vwredsum_i16m2 vs
// vsetvl_e8m2 / vle8_v_u8m2 / vwmacc_vv_i16m4 / vwredsum_i16m4). A capability FACT
// changes the lowering -- NOT a structural NULL.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

module {
  weft.exec.kernel @ggml_vec_dot_tq2_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          %sumi = weft_rvv.tq2_0_q8_k_ternary_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_tq2_0_q8_k_ternary_core", scale_model = "ternary-2bit-fused-plane-single-fp16-scale-i32-domain", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// ================= STAMPED ANCHOR (the SELECTION decision) ==================
// rv64gcv (VLEN128): the compiler SELECTED m2 (the ONLY anchor whose e8 VLMAX 32
// spans the 32-element plane at VLEN128) + the SEMANTIC minimum_vlen = 128 the
// verifier recomputes legality from. The gearbox now stamps the CONSTRUCTED brick.
// STAMP-VLEN128: weft_rvv.tq2_0_q8_k_ternary_core
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-NOT: weft_rvv.tq2_0_schedule.
//
// rv64gcv_zvl256b (VLEN256): the SAME brick FLIPS to m1 -- at VLEN256 m1's e8 VLMAX
// reaches 32, so it spans the plane in ONE vsetvl, TIES m2 on the capability-blind
// cost, and the lighter footprint breaks the tie to m1. The anchor MOVES with VLEN
// (the headline Win-A enrichment). minimum_vlen = 256 is stamped.
// STAMP-VLEN256: weft_rvv.tq2_0_q8_k_ternary_core
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

// ===================== VLEN128 (rv64gcv) — the m2 anchor ====================
// The compiler SELECTED m2: vsetvl_e16m4 acc, vsetvl_e8m2(32), vle8_v_u8m2 qs,
// vand/vsub i8m2 unpack, vle8_v_i8m2 q8, vwmacc_vv_i16m4, vwredsum_vs_i16m4_i32m1.
// VLEN128: emitc.func @weft_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(
// VLEN128: call_opaque "__riscv_vsetvl_e16m4"
// VLEN128: call_opaque "__riscv_vsetvl_e8m2"
// VLEN128: call_opaque "__riscv_vle8_v_u8m2"
// VLEN128: call_opaque "__riscv_vsub_vx_i8m2"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vwmacc_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vsetvl_e8m1"
// VLEN128-NOT: call_opaque "__riscv_vwmacc_vv_i16m2"
// VLEN128-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) — the FLIP ==================
// The compiler SELECTED m1: a BYTE-DIFFERENT kernel from the VLEN128 m2 shape. The
// accumulator narrows to i16m2 (vsetvl_e16m2), the strip to vsetvl_e8m1 (NOT e8m2),
// the qs load to u8m1, the q8 load to i8m1, the MAC to vwmacc_vv_i16m2, and the
// reduce to vwredsum_vs_i16m2_i32m1. This is the NON-NULL proof: the two VLENs do
// NOT emit the same bytes.
// VLEN256: emitc.func @weft_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(
// VLEN256: call_opaque "__riscv_vsetvl_e16m2"
// VLEN256: call_opaque "__riscv_vsetvl_e8m1"
// VLEN256: call_opaque "__riscv_vle8_v_u8m1"
// VLEN256: call_opaque "__riscv_vsub_vx_i8m1"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vwmacc_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// VLEN256-NOT: call_opaque "__riscv_vle8_v_u8m2"
// VLEN256-NOT: call_opaque "__riscv_vwmacc_vv_i16m4"
// VLEN256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN256: return
