// OPTION-2 STAGE B/C1 -- the C1 CLEANLINESS JUDGE: routing reads STRUCTURED
// OPPONENT FACTS, NOT the quant format LABEL. This fixture authors the abstract
// tcrv_rvv.quant_contraction op with NO `quant` attribute at all -- only the
// structured facts (block_dot_compute_heavy = true; no opponent_vlen_native_floor,
// i.e. no ggml VLEN-native opponent) plus the plain block-format facts and
// m_regime. At march=rv64gcv (Zvl128b => VLEN128) the fact-driven
// selectContractionAlgorithm still selects REPACK and the pass CONSTRUCTS the
// IDENTICAL typed tcrv_rvv.typed_repack_gemv_loop_body region (x16 facts 288/16/32,
// half_lanes 8, the two decomposed inner bricks, the DECLARED weight_layout_contract
// = "x16") it constructs when `quant = "q4_0"` is present -- byte-for-byte, same
// audit reason. This is the judge: delete the format string, keep the facts, and
// the compiler still constructs the right L2 region => quant is a label, routing
// is fact-driven (the measured win/loss knowledge lives in the IR, not in C++).
//
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        // NO `quant` attr -- routing decides from the structured facts alone.
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract op is GONE; the fact-driven selection realized the REPACK region --
// IDENTICAL to the quant="q4_0" VLEN128 cell in rvv-lower-quant-contraction-stage-
// b-selection.mlir (same x16 facts, same half_lanes, same audit reason).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.q4_0_q8_0_block_dot
// CHECK-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// CHECK: tcrv_rvv.typed_repack_gemv_loop_body
// CHECK-SAME: half_lanes = 8 : i64
// CHECK-SAME: tcrv_rvv.contraction_algorithm = "repack"
// CHECK-SAME: tcrv_rvv.path_materialization = "realized"
// CHECK-SAME: tcrv_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode"
// CHECK-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CHECK-SAME: weight_block_stride = 288 : i64
// CHECK-SAME: weight_interleave = 16 : i64
// CHECK-SAME: weight_quant_byte_offset = 32 : i64
// CHECK: tcrv_rvv.repack_lane_wise_q4_x_i8_dot
// CHECK: tcrv_rvv.repack_dual_fp16_scale_fold
// CHECK: tcrv_rvv.repack_dual_fp16_scale_fold
// CHECK: tcrv_rvv.typed_repack_gemv_loop_yield
