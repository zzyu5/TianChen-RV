module {
  weft.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %3 = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %4 = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %5 = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %6 = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %7 = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %8 = weft_rvv.setvl %0 {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %8 attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %3, %5, %1, %0, %2 attributes {activation_block_stride = 34 : i64, activation_quant_byte_offset = 2 : i64, fold_model = "lane_wise_vector_scale", half_lanes = 8 : i64, kind = "typed_repack_gemv_loop_body", qk = 32 : i64, scale_model = "dual-fp16-per-block-d_x.d_y", weft_rvv.contraction_algorithm = "repack", weft_rvv.path_materialization = "realized", weft_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode", weft_rvv.repack_accumulator_lmul_selection_reason = "capability-default-mf2", weft_rvv.weight_layout_contract = "x16", weight_block_stride = 352 : i64, weight_interleave = 16 : i64, weight_quant_byte_offset = 32 : i64} {
        ^bb0(%arg0: index, %arg1: !weft_rvv.vector<f32, "m2">, %arg2: !weft_rvv.vector<f32, "m2">):
          %9:2 = weft_rvv.repack_lane_wise_q4_x_i8_dot %3, %5, %8 block %arg0 : index {activation_quant_byte_offset = 2 : i64, kind = "repack_lane_wise_q4_x_i8_dot", weight_nibble_unsigned, weight_offset_bias = 16 : i64, weight_qh_byte_offset = 288 : i64, weight_quant_byte_offset = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          %10 = weft_rvv.repack_dual_fp16_scale_fold %3, %5, %9#0, %arg1, %8 block %arg0 : index {activation_scale_byte_offset = 0 : i64, kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          %11 = weft_rvv.repack_dual_fp16_scale_fold %3, %5, %9#1, %arg2, %8 block %arg0 : index {activation_scale_byte_offset = 0 : i64, kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %10, %11 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

