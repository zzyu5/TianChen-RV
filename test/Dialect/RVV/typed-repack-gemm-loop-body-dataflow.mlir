// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.typed_repack_gemm_loop_body region op makes the q4_0 16x1-REPACKED
// GEMM's inner contraction-block loop first-class in the typed RVV body -- the
// block-as-lane GEMM sibling of tcrv_rvv.typed_repack_gemv_loop_body. It carries the
// columnsPerPass per-column LANE-WISE f32 VECTOR accumulators (plus the block_index
// + runtime strip_row_offset region entry args), the ONE-strip N-column integer CORE
// brick (columnsPerPass per-column sumi), the columnsPerPass per-column dual-fp16
// scale FOLD bricks, and a yield naming the carried-out vectors. The verifier is
// fail-closed (I7) on wrong kind / fold_model / scale_model / block-format facts /
// ABI operand C type / region entry-arg count / brick result / yield accumulator
// count.

// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_loop_accepts_vlen128
module {
  tcrv.exec.kernel @repack_gemm_loop_accepts_vlen128 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_loop_accepts_vlen128", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.typed_repack_gemm_loop_body
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "lane_wise_vector_scale"} {
        ^bb0(%bi: index, %roff: index, %a0: !tcrv_rvv.vector<f32, "m2">, %a1: !tcrv_rvv.vector<f32, "m2">, %a2: !tcrv_rvv.vector<f32, "m2">, %a3: !tcrv_rvv.vector<f32, "m2">):
          %sumi:4 = tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %bi strip %roff : index, index {kind = "repack_gemm_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          %an0 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#0, %a0, %vl block %bi strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an1 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#1, %a1, %vl block %bi strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an2 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#2, %a2, %vl block %bi strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an3 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#3, %a3, %vl block %bi strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %an0, %an1, %an2, %an3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Fail-closed: the mf2 (absent integer_core_lmul) core folds all 4 columns per pass,
// so the region MUST carry columnsPerPass + 2 == 6 entry args. FIVE args (a dropped
// accumulator) is rejected.
module {
  tcrv.exec.kernel @repack_gemm_loop_rejects_wrong_argc {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_loop_rejects_wrong_argc", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires the region to carry exactly columnsPerPass + 2 (6) entry arguments}}
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "lane_wise_vector_scale"} {
        ^bb0(%bi: index, %roff: index, %a0: !tcrv_rvv.vector<f32, "m2">, %a1: !tcrv_rvv.vector<f32, "m2">, %a2: !tcrv_rvv.vector<f32, "m2">):
          %an0 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %a0, %a0, %vl block %bi strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %an0 : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Fail-closed: the loop op owns only fold_model "lane_wise_vector_scale"; any other
// fold tree is rejected.
module {
  tcrv.exec.kernel @repack_gemm_loop_rejects_wrong_fold {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_loop_rejects_wrong_fold", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{currently supports only fold_model "lane_wise_vector_scale"}}
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "scalar_scale"} {
        ^bb0(%bi: index, %roff: index, %a0: !tcrv_rvv.vector<f32, "m2">, %a1: !tcrv_rvv.vector<f32, "m2">, %a2: !tcrv_rvv.vector<f32, "m2">, %a3: !tcrv_rvv.vector<f32, "m2">):
          tcrv_rvv.typed_repack_gemm_loop_yield %a0, %a1, %a2, %a3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Fail-closed: the integer CORE brick owns only kind
// "repack_gemm_lane_wise_q4_x_i8_dot"; a foreign kind spelling is rejected.
module {
  tcrv.exec.kernel @repack_gemm_core_rejects_wrong_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bi = tcrv_rvv.runtime_abi_value {c_name = "bi", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bi", role = "runtime-element-count"} : index
      %roff = tcrv_rvv.runtime_abi_value {c_name = "roff", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "roff", role = "source-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_core_rejects_wrong_kind", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{currently supports only kind "repack_gemm_lane_wise_q4_x_i8_dot"}}
        %sumi:4 = tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %bi strip %roff : index, index {kind = "repack_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
      } : !tcrv_rvv.vl
    }
  }
}
