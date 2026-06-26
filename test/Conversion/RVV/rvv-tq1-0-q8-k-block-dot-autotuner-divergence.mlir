// The tq1_0 Win-A proof: the COMPILER SELECTS the ggml TQ1_0 x Q8_K integer-DOT
// anchor (the base-3 unpack is fixed), and the selection DIVERGES by capability
// from the SAME attr-less input -- via the UNIFIED schedule autotuner (the SAME
// walk-all pass that auto-discovers every TunableScheduleOpInterface op).
//
// The kernel below carries NO integer_core_lmul knob -- the compiler must compute
// it. tq1_0's base-3 trit unpack (section A) lands an element-ordered aux8[256];
// the integer dot (section B) is a flat 256-element aux8 x q8 contraction (single
// fp16 scale, NO per-sub-block scale), so it widens to 32-lane strips. The single
// vsetvl_e8<anchor>(32) cover is correct ONLY at the whole-LMUL anchor whose i8
// strip VLMAX spans 32 at the derived minimum VLEN. WHICH anchor that is MOVES
// with VLEN:
//
//   * at VLEN 128: only m2 spans it (e8m1 VLMAX 16 < 32) -> integer_core_lmul "m2"
//     (dot widens i16m4).
//   * at VLEN 256: m1 also reaches 32; the lighter footprint breaks the tie ->
//     integer_core_lmul "m1" (dot i16m2).
//
// One capability fact (the REAL VLEN bits) -> the dot anchor FLIPS m2->m1. NOTE:
// this widens ONLY the dot; the base-3 unpack + aux8 round-trip are KEPT (ggml's
// tq1_0 _vl128/_vl256 are DIFFERENT shapes, not a clean LMUL flip -- full fusion
// is a separate larger emit), so this is an honest below-parity LOSS reduction.

// First, the DECISION-LEVEL proof: the unified autotuner stamps DIFFERENT anchors
// onto the SAME attr-less op purely by the VLEN capability fact (no lowering).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Then the EMISSION-LEVEL non-NULL proof: VLEN256 emits a BYTE-DIFFERENT dot from
// VLEN128 (vwmul_vv_i16m2 / vwredsum_i16m2 vs vwmul_vv_i16m4 / vwredsum_i16m4). A
// capability FACT changes the lowering -- NOT a structural NULL.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

module {
  tcrv.exec.kernel @ggml_vec_dot_tq1_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.tq1_0_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_tq1_0_q8_k_block_dot", scale_model = "ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 48 : i64, weight_d_byte_offset = 52 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ================= STAMPED ANCHOR (the SELECTION decision) ==================
// rv64gcv (VLEN128): the compiler SELECTED m2 (the ONLY anchor whose e8 VLMAX 32
// spans the 32-lane dot strip at VLEN128) + the SEMANTIC minimum_vlen = 128.
// STAMP-VLEN128: tcrv_rvv.tq1_0_q8_k_block_dot
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.tq1_0_schedule.has_zvl128b = true
// STAMP-VLEN128-SAME: tcrv_rvv.tq1_0_schedule.producer = "rvv-tq1-0-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME op FLIPS to m1 -- at VLEN256 m1's e8 VLMAX
// reaches 32, spans the strip in ONE vsetvl, and the lighter footprint breaks the
// tie to m1. minimum_vlen = 256 is stamped.
// STAMP-VLEN256: tcrv_rvv.tq1_0_q8_k_block_dot
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

// ===================== VLEN128 (rv64gcv) — the m2 dot anchor ================
// The compiler SELECTED m2: the integer dot widens to vle8_v_i8m2 + vwmul_vv_i16m4
// + vwredsum_vs_i16m4_i32m1 (the base-3 unpack -- vwmulu_vx_u16m4 etc -- is fixed).
// VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(
// VLEN128: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN128-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) — the FLIP =================
// The compiler SELECTED m1: a BYTE-DIFFERENT dot from the VLEN128 m2 shape. The
// dot MAC narrows to vwmul_vv_i16m2 and the reduce to vwredsum_vs_i16m2_i32m1.
// This is the NON-NULL proof: the two VLENs do NOT emit the same dot bytes.
// VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(
// VLEN256: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN256: return
