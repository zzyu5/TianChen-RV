// The iq2_xxs Win-A proof (PRESERVED across the iq2_xxs flip): the COMPILER SELECTS the
// ggml IQ2_XXS x Q8_K GRID-codebook integer-core anchor, and the selection DIVERGES by
// capability from the SAME attr-less input -- via the UNIFIED schedule autotuner (the
// SAME walk-all pass that auto-discovers every TunableScheduleOpInterface op, NO
// per-iq2_xxs pass). The monolith op was RETIRED at the flip; the Win-A gearbox moved
// verbatim onto the CONSTRUCTED tcrv_rvv.iq2_xxs_q8_k_grid_core brick (SAME kernel key
// "iq2_xxs"), so the autotuner stamps the SAME m2->m1 selection onto the brick with NO
// registry change. This test now drives the CONSTRUCTED typed super-block
// SCALAR-accumulator GRID loop body (fold_model "scalar_delta_grid"), not the retired
// monolith op.
//
// The grid-core brick below carries NO integer_core_lmul knob -- the compiler must
// compute it. iq2_xxs's grid+sign vluxei16 gather + dot is ALWAYS one 32-lane sub-block
// body (gather 4 u64 grid + 4 u64 sign entries as i64<anchor>, reinterpret to i8<anchor>,
// fold the per-lane sign, ONE vwmul_vv_i16<2*anchor> + ONE vwredsum per sub-block). The
// single vsetvl/vluxei16_v_i64<anchor> cover is correct ONLY at the whole-LMUL anchor
// whose i8 strip VLMAX spans the 32-element sub-block at the derived minimum VLEN (a
// single i64 is 8 bytes, so the 4-entry gather needs the i64 anchor whose VLMAX reaches 4
// = the i8 view reaches 32). WHICH anchor that is MOVES with VLEN:
//
//   * at VLEN 128: only m2 spans it (e8m1 VLMAX 16 < 32) -> integer_core_lmul "m2"
//     (i64m2 gather, i8m2 view, wide accumulator i16m4, u16mf2 index).
//   * at VLEN 256: m1 also reaches 32 (and i64m1 VLMAX reaches the 4 grid entries);
//     m1 TIES m2 on the capability-blind cost and the lighter footprint breaks the
//     tie -> integer_core_lmul "m1" (i64m1 gather, i8m1 view, i16m2 acc, u16mf4
//     index) -- exactly the m1-32-lane shape ggml's shipped _vl256 kernel uses.
//
// One capability fact (the REAL VLEN bits) -> the anchor FLIPS m2->m1. This is the
// compiler SELECTING the shape from a capability fact, not a hand-set attr.

// First, the DECISION-LEVEL proof: the unified autotuner stamps DIFFERENT anchors onto
// the SAME attr-less grid-core brick purely by the VLEN capability fact (no lowering).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Then the EMISSION-LEVEL non-NULL proof: VLEN256 emits a BYTE-DIFFERENT kernel from
// VLEN128 (vluxei16_v_i64m1 / vreinterpret i8m1 / vmul_vv_i8m1 / vle8_v_i8m1 /
// vwmul_vv_i16m2 / vwredsum_i16m2 / vle16_v_u16mf4 vs the m2/i16m4/mf2 shape). A
// capability FACT changes the lowering -- NOT a structural NULL.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

module {
  tcrv.exec.kernel @ggml_vec_dot_iq2_xxs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2xxs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq2_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq2_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          %bsum = tcrv_rvv.iq2_xxs_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq2_xxs_q8_k_grid_core", scale_model = "per-sub-block-aux1-scale-grid-of-8-codebook-signs64-sign-plane-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// ================= STAMPED ANCHOR (the SELECTION decision) ==================
// rv64gcv (VLEN128): the compiler SELECTED m2 (the ONLY anchor whose e8 VLMAX 32 spans
// the 32-element sub-block at VLEN128) + the SEMANTIC minimum_vlen = 128 the verifier
// recomputes legality from. The gearbox now stamps the CONSTRUCTED grid-core brick.
// STAMP-VLEN128: tcrv_rvv.iq2_xxs_q8_k_grid_core
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.iq2_xxs_schedule.has_zvl128b = true
// STAMP-VLEN128-SAME: tcrv_rvv.iq2_xxs_schedule.producer = "rvv-iq2-xxs-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME brick FLIPS to m1 -- at VLEN256 m1's e8 VLMAX
// reaches 32 and i64m1 VLMAX reaches the 4 grid entries, so it spans the sub-block in ONE
// gather, TIES m2 on the capability-blind cost, and the lighter footprint breaks the tie
// to m1. The anchor MOVES with VLEN (the headline Win-A enrichment, matching ggml's
// _vl256 m1-32-lane shape). minimum_vlen = 256 is stamped.
// STAMP-VLEN256: tcrv_rvv.iq2_xxs_q8_k_grid_core
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

// ===================== VLEN128 (rv64gcv) — the m2 anchor ====================
// The compiler SELECTED m2: vluxei16_v_i64m2 grid+sign gather, vreinterpret i8m2,
// vmul_vv_i8m2 sign fold, vle8_v_i8m2 q8, vwmul_vv_i16m4, vwredsum_vs_i16m4_i32m1,
// vle16_v_u16mf2 index.
// VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
// VLEN128: call_opaque "__riscv_vle16_v_u16mf2"
// VLEN128: call_opaque "__riscv_vluxei16_v_i64m2"
// VLEN128: call_opaque "__riscv_vreinterpret_v_i64m2_i8m2"
// VLEN128: call_opaque "__riscv_vmul_vv_i8m2"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vluxei16_v_i64m1"
// VLEN128-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN128-NOT: call_opaque "__riscv_vle16_v_u16mf4"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) — the FLIP ==================
// The compiler SELECTED m1: a BYTE-DIFFERENT kernel from the VLEN128 m2 shape. The gather
// narrows to i64m1, the view to i8m1, the sign fold to vmul_vv_i8m1, the q8 load to i8m1,
// the product to vwmul_vv_i16m2, the reduce to vwredsum_vs_i16m2_i32m1, and the index load
// to u16mf4. This is the NON-NULL proof: the two VLENs do NOT emit the same bytes.
// VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
// VLEN256: call_opaque "__riscv_vle16_v_u16mf4"
// VLEN256: call_opaque "__riscv_vluxei16_v_i64m1"
// VLEN256: call_opaque "__riscv_vreinterpret_v_i64m1_i8m1"
// VLEN256: call_opaque "__riscv_vmul_vv_i8m1"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vluxei16_v_i64m2"
// VLEN256-NOT: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN256-NOT: call_opaque "__riscv_vle16_v_u16mf2"
// VLEN256: return
