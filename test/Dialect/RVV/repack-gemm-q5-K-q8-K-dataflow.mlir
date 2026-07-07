// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.repack_gemm_q5_K_q8_K op makes the ggml q5_K 16x1-REPACKED PREFILL
// GEMM hot kernel first-class in the typed RVV body -- q5_K == q4_K + the qh
// 5th-bit plane, the prefill sibling of the q5_K GEVM. It consumes the repacked
// block_q5_Kx16 weights (vx, byte stride (n/QK_K)*2816) + the INTERLEAVED
// block_q8_Kx4 activation stream (vy, stride 1168) + the fp32 output (s) + n + nr
// + nc + bs + vl, and accumulates LANE-WISE via vwmacc (NO cross-lane vredsum).
// It carries the K-quant super-block facts PLUS the q5_K qh high-bit-plane offset
// (qh at +256, qs at +768) as typed attrs (I4 mirror). The verifier is
// fail-closed (I7) on a wrong kind / scale model / repacked block-format fact --
// INCLUDING the q5_K-specific qh byte offset -- / strip width / core anchor.

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q5k_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemm_q5k_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_repack_gemm_q5_K_q8_K"}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "gemm", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 2816}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2304 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_wrong_qs_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_wrong_qs_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_quant_byte_offset == 768}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_wrong_qh_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_wrong_qh_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_qh_byte_offset == 256}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 768 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5"}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q5k_accepts_half16
  tcrv.exec.kernel @repack_gemm_q5k_accepts_half16 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_accepts_half16", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_half12 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_half12", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires half_lanes in {8, 16}}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 12 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q5k_accepts_m1_half16
  tcrv.exec.kernel @repack_gemm_q5k_accepts_m1_half16 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_accepts_m1_half16", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_q5k_rejects_m1_half8 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q5k_rejects_m1_half8", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires half_lanes == 16 when integer_core_lmul is "m1"}}
        %g = tcrv_rvv.repack_gemm_q5_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
