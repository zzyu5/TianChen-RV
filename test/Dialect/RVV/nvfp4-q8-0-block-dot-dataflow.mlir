// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.nvfp4_q8_0_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_nvfp4_q8_0 block dot-product STRUCTURE first-class in the typed RVV
// body -- the SECOND FP4-class sibling (NVIDIA's FP4). It REUSES mxfp4's e2m1
// kvalues_mxfp4 codebook gather, but carries a per-SUB-block UE4M3 fp8 scale on a
// QK=64 SUPER-block: block_nvfp4 = { uint8_t d[4]; uint8_t qs[32] } (stride 36, four
// UE4M3 scales then the 32 packed FP4 nibble bytes at +4), spanning TWO block_q8_0
// activation blocks (sub-block s -> q8 block 2*ib+s/2, half-offset (s%2)*16). It
// consumes the nvfp4 weight base (vx), the q8_0 activation base (vy), the fp32 output
// (s), the runtime element count (n), and the active vl token, and carries the
// super-block-format structural facts (qk, qk_sub, strides, the dual quant byte
// offsets, the per-sub-block q8 high-half offset) PLUS the 16-entry codebook as typed
// attrs (I4 mirror). The verifier is fail-closed (I7) on a wrong kind / scale model /
// super-block-format fact / codebook size / non-m1 anchor / operand C type.

// CHECK-LABEL: tcrv.exec.kernel @nvfp4_q8_0_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.nvfp4_q8_0_block_dot
        // CHECK-SAME: codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_nvfp4_q8_0_block_dot"}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong scale model (must be the UE4M3 half-per-sub-block model; the full
// decode without *0.5f -- which would double every result -- is fail-closed, I7).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "ue4m3-half-per-sub-block"}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-full-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong super-block stride (block_nvfp4 = 4 UE4M3 bytes + 32 FP4 nibble
// bytes = 36; mxfp4's 17 would mis-stride the super-block).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 36}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight quant byte offset (the FP4 nibbles must be at +4, after the
// four UE4M3 sub-block scale bytes -- NOT mxfp4's +1, which would read into a scale).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_weight_quant_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_weight_quant_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_quant_byte_offset == 4}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong per-sub-block q8 high-half offset (must be 8 = QK_NVFP4_SUB/2, NOT
// mxfp4's 16, which is the wrong sub-block geometry).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_high_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_high_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires activation_high_byte_offset == 8}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a codebook with the wrong number of entries (must be EXACTLY 16, one per
// FP4 nibble index [0,15]). A 15-entry table cannot index the nibbles -> fail-closed.
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_codebook_size {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_codebook_size", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires codebook to carry exactly 16 int8 entries}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a non-m1 integer-core anchor (the 16-entry codebook gather needs the
// broadcast table register's VLMAX >= 16, which mf4 cannot provide at VLEN=128).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_non_m1_anchor {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_non_m1_anchor", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts integer_core_lmul "m1"}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "mf4"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type (the *s destination must be float *).
module {
  tcrv.exec.kernel @nvfp4_q8_0_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "nvfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "nvfp4_q8_0_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.nvfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_nvfp4_q8_0_block_dot", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
