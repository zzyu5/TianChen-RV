// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.repack_gemv_q3_K_q8_K op makes the ggml q3_K 16x1-REPACKED GEVM
// (decode) hot kernel first-class in the typed RVV body -- q3_K is q6_K's no-min
// structural cousin: a 3-bit SIGNED weight assembled from a 2-bit qs low plane + a
// SINGLE 1-bit hmask high plane, SUBTRACTIVE (bit CLEAR -> subtract 4;
// `((qs&3)|(hbit<<2)) - 4`), 16 SIGNED 6-bit per-sub-block scales (pre-unpacked +
// -32-biased at repack), and NO per-sub-block min term (a SINGLE accumulator, the -4
// bias inside each weight lane). It consumes the repacked block_q3_Kx16 weights (vx,
// byte stride (n/QK_K)*1824) + ONE plain block_q8_K activation stream (vy, stride 292)
// + the fp32 output (s) + n + nc + vl, and accumulates LANE-WISE via vwmacc (NO
// cross-lane vredsum). It carries the q3_K super-block facts (qs at +800, hmask at
// +288, signed scales at +32, 16 sub-blocks) as typed attrs (I4 mirror) and REJECTS
// the q4_K/q5_K dmin/bsums offsets (no-min structure). The verifier is fail-closed (I7)
// on a wrong kind / scale model / repacked block-format fact / strip width / core
// anchor / a forbidden dmin/bsums attribute.

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemv_q3k_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemv_q3k_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_repack_gemv_q3_K_q8_K"}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "gemv", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 1824}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1568 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_wrong_qs_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_wrong_qs_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_quant_byte_offset == 800}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  // The q3_K SUBTRACTIVE-hmask high-bit plane offset is pinned fail-closed: a wrong
  // weight_hmask_byte_offset (the high bit the SUBTRACTIVE assembly reads) is rejected.
  tcrv.exec.kernel @repack_gemv_q3k_rejects_wrong_hmask_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_wrong_hmask_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_hmask_byte_offset == 288}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 256 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_wrong_scales_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_wrong_scales_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_scales_byte_offset == 32}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 64 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin"}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  // The q3_K no-min structure is ENFORCED fail-closed: a stray q4_K/q5_K
  // weight_dmin_byte_offset attribute (the min-term scale q3_K does NOT have, like
  // q6_K) is rejected as an unexpected attribute.
  tcrv.exec.kernel @repack_gemv_q3k_rejects_dmin_attr {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_dmin_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts the bounded ggml Q3_K x Q8_K 16x1-repacked GEMV attributes}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, weight_dmin_byte_offset = 32 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemv_q3k_accepts_half16
  tcrv.exec.kernel @repack_gemv_q3k_accepts_half16 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_accepts_half16", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.repack_gem
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_q3k_rejects_m1_half8 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q3k_rejects_m1_half8", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires half_lanes == 16 when integer_core_lmul is "m1"}}
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "m1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
