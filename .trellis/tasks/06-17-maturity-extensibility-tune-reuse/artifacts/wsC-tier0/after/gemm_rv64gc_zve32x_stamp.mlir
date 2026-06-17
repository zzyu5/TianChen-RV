module {
  tcrv.exec.kernel @gemm {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @gemm attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %5 = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %6 = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %7 = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %8 = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %9 = tcrv_rvv.setvl %0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %9 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @gemm, sew = 32 : i64, source_kernel = "gemm", status = "selected-lowering-boundary"} {
        %10 = tcrv_rvv.q4_0_q8_0_gemm %2, %3, %4, %1, %0, %5, %6, %7, %8, %9 {activation_block_stride = 34 : i64, activation_cols = 4 : i64, activation_high_byte_offset = 16 : i64, kind = "ggml_q4_0_q8_0_gemm", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_model = "dual-fp16-per-block-d_x.d_y", tcrv_rvv.q4_0_gemm_schedule.candidate_count = 5 : i64, tcrv_rvv.q4_0_gemm_schedule.has_zvl128b = false, tcrv_rvv.q4_0_gemm_schedule.legal_candidate_count = 5 : i64, tcrv_rvv.q4_0_gemm_schedule.producer = "rvv-gemm-m-autotuner", tcrv_rvv.q4_0_gemm_schedule.selected_cost = 0 : i64, tcrv_rvv.q4_0_gemm_schedule.selection_reason = "static default GEMM M-block (the safe cache-friendly tile; no tuning record -> the measurement-tuned M is not available)", tcrv_rvv.q4_0_gemm_schedule.vreg_ceiling = 8 : i64, weight_block_stride = 18 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.runtime_abi_value, index, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

