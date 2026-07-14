// OPTION-2 STAGE B/C1 -- [G8 六.3] the PER-FORMAT MEASURED VLEN256-DECODE cell of the
// in-compiler contraction-path SELECTION. This fixture is the in-IR proof that the old
// BLANKET fact-3 rule ("VLEN256 decode always declines repack", which generalized a
// q4_0-only 0.74x LOSS) is now a PER-FORMAT BOARD-MEASURED decision: the same
// (VLEN256, decode) cell that DECLINES for q4_0/iq4_nl (measured-negative) now SELECTS
// repack for q5_0 (measured-BENEFICIAL 1.190x on k1, casefile
// experiments/active/g8-stage3-attack/k1-gevm-sweep, commit ac5ea76f). The decision is
// driven by the board-seeded RVVLowerQuantContraction kRepackVlen256DecodeMeasurements
// registry keyed on the committed decode-family scale_model WHAT; the pure selector
// selectContractionAlgorithm stays BLIND to the format label and reads only the derived
// fact (vlen256DecodeRepackBeneficial). NO perf/e2e claim -- lit-emitted, NOT run.
//
// The SAME q5_0-decode module is lowered at TWO -march tiers to prove the fix is scoped
// to the VLEN256-DECODE cell ONLY (rvv/VLEN128 sees ZERO drift):
//
// VLEN256 (rv64gcv_zvl256b => 256, the k1 decode cell): q5_0 decode -> REPACK SELECTED
// via the board-MEASURED-BENEFICIAL fact -> REALIZED as the typed
// weft_rvv.typed_repack_gemv_loop_body region (half_lanes 16), reason names the measured
// cell. This is the CELL the blanket rule wrongly declined.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=VLEN256
//
// VLEN128 (rv64gcv => 128, the rvv deployed decode cell): UNCHANGED -- repack via the
// capability/regime rule (half_lanes 8), the q4_0-vlen128 audit token. Proves the fix
// did NOT touch the VLEN128 decode path (zero rvv drift).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128

module {
  weft.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q5_0", scale_model = "dual-fp16-per-block-d_x.d_y-five-bit", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (VLEN256 tier) the board-MEASURED-BENEFICIAL fact SELECTS repack: the abstract op is
// GONE, no block-dot op is emitted (the blanket rule would have declined to block-dot),
// and the typed repack-GEVM region is realized with the VLEN256 half_lanes 16 strip +
// the x16 contract. The audit reason NAMES the measured cell (format-blind).
// VLEN256-NOT: weft_rvv.quant_contraction
// VLEN256-NOT: weft_rvv.q5_0_q8_0_block_dot
// VLEN256: weft_rvv.typed_repack_gemv_loop_body
// VLEN256-SAME: half_lanes = 16 : i64
// VLEN256-SAME: weft_rvv.contraction_algorithm = "repack"
// VLEN256-SAME: weft_rvv.path_materialization = "realized"
// VLEN256-SAME: weft_rvv.path_selection_reason = "repack-kept-vlen256-decode-measured-beneficial"
// VLEN256-SAME: weft_rvv.weight_layout_contract = "x16"
// VLEN256-SAME: weight_interleave = 16 : i64
// VLEN256: weft_rvv.repack_lane_wise_q4_x_i8_dot
// VLEN256: weft_rvv.repack_dual_fp16_scale_fold
// VLEN256: weft_rvv.typed_repack_gemv_loop_yield

// (VLEN128 tier) UNCHANGED: repack via the capability/regime rule (fact 3 minVLEN==128),
// half_lanes 8, the historical q4_0-vlen128 audit token -- proving the per-format
// VLEN256-decode fix left the rvv/VLEN128 decode path byte-identical (zero drift).
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128-NOT: weft_rvv.q5_0_q8_0_block_dot
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 8 : i64
// VLEN128-SAME: weft_rvv.contraction_algorithm = "repack"
// VLEN128-SAME: weft_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode"
