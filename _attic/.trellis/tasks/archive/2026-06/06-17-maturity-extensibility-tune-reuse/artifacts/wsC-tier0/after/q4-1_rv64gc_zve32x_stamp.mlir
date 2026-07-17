module {
  tcrv.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.setvl %0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %4 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %5 = tcrv_rvv.q4_1_q8_1_block_dot %2, %3, %1, %0, %4 {activation_block_stride = 36 : i64, activation_high_byte_offset = 16 : i64, activation_sum_byte_offset = 2 : i64, integer_core_lmul = "m1", kind = "ggml_q4_1_q8_1_block_dot", multi_block_factor = 2 : i64, qk = 32 : i64, quant_byte_offset = 4 : i64, scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", strip_elision = "robust", tcrv_rvv.q4_1_schedule.candidate_count = 12 : i64, tcrv_rvv.q4_1_schedule.has_zvl128b = false, tcrv_rvv.q4_1_schedule.legal_candidate_count = 6 : i64, tcrv_rvv.q4_1_schedule.peak_live_vector_registers = 6 : i64, tcrv_rvv.q4_1_schedule.producer = "rvv-q4-1-autotuner", tcrv_rvv.q4_1_schedule.selected_cost = 1310 : i64, tcrv_rvv.q4_1_schedule.selection_reason = "min-cost legal Q4_1 shape (capability-blind structural cost; strip-elision gated on Zvl128b)", tcrv_rvv.q4_1_schedule.vector_register_budget = 32 : i64, weight_block_stride = 20 : i64, weight_min_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

