// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.repack_gemm_q3_K_q8_K op makes the ggml q3_K 16x1-REPACKED PREFILL GEMM
// (M>>1) hot kernel first-class in the typed RVV body -- the prefill sibling of
// tcrv_rvv.repack_gemv_q3_K_q8_K. Same 3-bit SUBTRACTIVE-hmask signed-weight
// (`((qs&3)|(hbit<<2)) - 4`) + 16 SIGNED 6-bit scale + no-min structure, with the
// interleaved block_q8_Kx4 activation (stride 1168, qs at +16). It consumes the
// repacked block_q3_Kx16 weights (vx, stride 1824) + the interleaved q8_Kx4 activation
// (vy) + the fp32 output (s) + n + nr + nc + bs + vl. It carries weight_hmask_byte_
// offset (NOT weight_qh_byte_offset) and REJECTS the q4_K/q5_K dmin/bsums offsets. The
// verifier is fail-closed (I7) on a wrong kind / scale model / block-format fact /
// strip width / core anchor / a forbidden dmin/bsums attribute.

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q3k_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemm_q3k_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q3k_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_repack_gemm_q3_K_q8_K"}}
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "gemm", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q3k_rejects_wrong_act_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_rejects_wrong_act_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires activation_block_stride == 1168}}
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  // The q3_K SUBTRACTIVE-hmask high-bit plane offset is pinned fail-closed on the GEMM
  // route too.
  tcrv.exec.kernel @repack_gemm_q3k_rejects_wrong_hmask_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_rejects_wrong_hmask_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_hmask_byte_offset == 288}}
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 320 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  // The q3_K no-min structure is ENFORCED fail-closed on the GEMM route: a stray
  // activation_bsums_byte_offset attribute (the min-term q3_K does NOT have) is
  // rejected as an unexpected attribute.
  tcrv.exec.kernel @repack_gemm_q3k_rejects_bsums_attr {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_rejects_bsums_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts the bounded ggml Q3_K x Q8_K 16x1-repacked GEMM attributes}}
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q3k_accepts_half16
  tcrv.exec.kernel @repack_gemm_q3k_accepts_half16 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_accepts_half16", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemm_q3_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
