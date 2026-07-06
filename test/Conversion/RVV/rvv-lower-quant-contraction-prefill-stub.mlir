// OPTION-2 STAGE B + C1 -- the q4_0 PREFILL cell of the IN-COMPILER
// contraction-path SELECTION (B) and the C1 in-IR BRIDGE. The selector's fact 3
// keeps Repack for ANY prefill regime, M-amortized, INDEPENDENT of VLEN -- so
// q4_0 prefill selects Repack even at the default -march "" (VLEN 0). C1 then
// REALIZES that Repack selection as a real repack op ONLY where the capability
// affords an e16m1 strip width (deriveRepackHalfLanes(minVLEN) in {8,16}, i.e.
// minVLEN >= 128); at VLEN0 there is no strip width, so the bridge keeps the
// deferred block-dot stub. This fixture proves BOTH prefill arms:
//
// DEFERRED (VLEN0): prefill Repack-SELECTED but no capability strip width ->
// deferred block-dot stub (byte-identical to stage B; materialization deferred).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction | FileCheck %s --check-prefix=DEFERRED
//
// REALIZED (VLEN128, rv64gcv => Zvl128b => 128): prefill Repack-SELECTED AND a
// capability strip width (half_lanes 8) -> REALIZED as the real repack op +
// weight_layout_contract = "x16", with the PREFILL reason token (distinct from
// the decode reason in the stage-b-selection fixture).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=REALIZED
//
// LATENT-MISPICK HONESTY: fact 3 keeps Repack for ANY prefill, so q4_0 @
// K1-VLEN256 *prefill* is auto-kept though only the *decode* cell was measured.
// This is a known latent mispick recorded in the design, not a validated cell.

module {
  tcrv.exec.kernel @quant_contraction_prefill_select {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "quant_contraction_prefill_select", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// DEFERRED (VLEN0): the abstract op is lowered; Repack was SELECTED but with no
// capability strip width it CANNOT form an x16 repack op, so the deferred
// block-dot stub is kept (materialization deferred to stage C/system).
// DEFERRED-NOT: tcrv_rvv.quant_contraction
// DEFERRED-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// DEFERRED: tcrv_rvv.q4_0_q8_0_block_dot
// DEFERRED-SAME: tcrv_rvv.contraction_algorithm = "repack"
// DEFERRED-SAME: tcrv_rvv.path_materialization = "deferred-stage-c"
// DEFERRED-SAME: tcrv_rvv.path_selection_reason = "repack-kept-q4_0-prefill"

// REALIZED (VLEN128): the prefill Repack selection is CONSTRUCTED as the typed
// tcrv_rvv.typed_repack_gemm_loop_body REGION (the block-as-lane GEMM the PREFILL
// m_regime selects, distinct from the decode GEVM). It carries the x16 weight facts
// (stride 288, half_lanes 8) AND the block_q8_0x4 INTERLEAVED activation facts
// (stride 136, activation_interleave 4) + the OUTPUT CONTRACT weight_layout_contract
// = "x16", with the PREFILL reason token, plus the two decomposed inner GEMM bricks
// + yield. The bridge MATERIALIZES the two GEMM ABI values (row count nr, output row
// stride bs) the abstract op does not carry. This proves the prefill GEMM
// realization arm; the decode arm (GEVM) is proven in the stage-b-selection fixture.
// REALIZED-NOT: tcrv_rvv.quant_contraction
// REALIZED-NOT: tcrv_rvv.q4_0_q8_0_block_dot
// REALIZED-NOT: tcrv_rvv.repack_gemm_q4_0_q8_0
// REALIZED-NOT: tcrv_rvv.typed_repack_gemv_loop_body
// The two materialized GEMM ABI values (nr row count, bs output row stride).
// REALIZED: tcrv_rvv.runtime_abi_value {c_name = "nr"
// REALIZED: tcrv_rvv.runtime_abi_value {c_name = "bs"
// REALIZED: tcrv_rvv.typed_repack_gemm_loop_body
// REALIZED-SAME: activation_block_stride = 136 : i64
// REALIZED-SAME: activation_interleave = 4 : i64
// REALIZED-SAME: half_lanes = 8 : i64
// REALIZED-SAME: tcrv_rvv.contraction_algorithm = "repack"
// REALIZED-SAME: tcrv_rvv.path_materialization = "realized"
// REALIZED-SAME: tcrv_rvv.path_selection_reason = "repack-kept-q4_0-prefill"
// REALIZED-SAME: tcrv_rvv.weight_layout_contract = "x16"
// REALIZED-SAME: weight_block_stride = 288 : i64
// REALIZED: tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot
// REALIZED: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// REALIZED: tcrv_rvv.typed_repack_gemm_loop_yield
