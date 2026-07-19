// [GAP-P1]-loosen JUDGMENT -- the per-format repack accumulator-LMUL measured gate.
// The front-door selector selectRepackAccumulatorLMUL consults the board-MEASURED
// table lookupRepackMeasuredM1Faster (registration-as-DATA, keyed on the committed
// decode-family scale_model WHAT). Board-measured rows @rvv VLEN128 (spill-free,
// byte-exact, 2-seed cold):
//   * q8_0 full-i8 (kNibbleQ80ScaleModel)         -> m1 (GEVM 1.4-2.4x / GEMM 1.40x)
//   * q4_0 signed-nibble (kNibbleQ40ScaleModel)   -> m1 (GEVM 2.3-2.5x / GEMM 1.24x)
//   * q4_1 nibble+min (kNibbleQ41ScaleModel)      -> m1 (GEVM 2.3x / GEMM parity)
//       (see experiments/active/r51f-gapp1-q8-deployed-board/FINDING.md +
//        experiments/active/r51g-b1-repack-family-sweep/FINDING.md)
// Every OTHER format has NO row => nullopt => the mf2 default holds (no-blind-widest).
// q5_0/q5_1 (+qh) stay mf2 ON PURPOSE: their m1 wins the decode GEVM but the prefill
// GEMM m1 core SPILLS and runs 2.3-2.4x SLOWER -- [GAP-P1]'s regfile-spill concern,
// board-vindicated; one measured row flips BOTH regimes, so they are NOT flipped.
//
// This fixture is the DISCRIMINATOR: the SAME pass, SAME board (rv64gcv => VLEN128),
// SAME decode repack-GEVM construction path, differing ONLY in the format input:
//   * q8_0  (full-i8)         -> WIDE m1 chain (half_lanes 16, integer_core_lmul m1,
//                                reason "measured").
//   * q5_1  (nibble+min+qh)   -> mf2 default  (half_lanes 8,  integer_core_lmul mf2,
//                                reason "capability-default-mf2").
// Flip the format input and the accumulator LMUL flips; the measured gate is a REAL
// per-format consumer, not a blanket widen.
//
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s

module {
  // ---- q8_0 (full-i8): board-measured => WIDE m1 -------------------------------
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q8_0", scale_model = "dual-fp16-per-block-d_x.d_y-full-i8", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_memory_bound = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
  // ---- q5_1 (nibble+min+qh): NO measured row => mf2 default (spill-vindicated) --
  weft.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q5_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min-five-bit", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 24 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// q8_0 full-i8: the board-MEASURED row flips the accumulator to the WIDE m1 chain.
// CHECK: weft_rvv.typed_repack_gemv_loop_body
// CHECK-SAME: half_lanes = 16 : i64
// CHECK-SAME: integer_core_lmul = "m1"
// CHECK-SAME: weft_rvv.repack_accumulator_lmul_selection_reason = "measured"

// q5_1 nibble+min+qh: NO measured row => the mf2 default holds (no-blind-widest;
// the prefill GEMM m1 core spills, so it is NOT flipped). The SAME construction
// path, ONLY the format differs -- and the accumulator does NOT widen.
// CHECK: weft_rvv.typed_repack_gemv_loop_body
// CHECK-SAME: half_lanes = 8 : i64
// CHECK-SAME: integer_core_lmul = "mf2"
// CHECK-SAME: weft_rvv.repack_accumulator_lmul_selection_reason = "capability-default-mf2"
