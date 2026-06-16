// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.mxfp4_q8_0_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_mxfp4_q8_0 block dot-product STRUCTURE first-class in the typed RVV
// body -- the CODEBOOK-class sibling that opens the LAST structural quantization
// class: FP4 (e2m1) weights with an E8M0 shared-exponent block scale. It is the AoS
// QK=32 block loop, the per-block E8M0 weight scale 2^(e-128) (the HALF form), the
// q8_0 fp16 scale, the NON-LINEAR FP4 codebook decode (each 4-bit nibble indexes the
// 16-entry kvalues_mxfp4[16] int8 lookup table), the fp32 scalar accumulation, and
// the *s store. block_mxfp4 = { uint8_t e; uint8_t qs[16] } (stride 17, nibbles @1).
// It consumes the mxfp4 weight base (vx), the q8_0 activation base (vy), the fp32
// output (s), the runtime element count (n), and the active vl token, and carries the
// block-format structural facts (strides, dual quant byte offsets, q8 high-half
// offset) PLUS the 16-entry codebook as typed attrs (I4 mirror). The verifier is
// fail-closed (I7) on a wrong kind / scale model / block-format fact / codebook size /
// non-m1 anchor / operand C type / shape knob.

// CHECK-LABEL: tcrv.exec.kernel @mxfp4_q8_0_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.mxfp4_q8_0_block_dot
        // CHECK-SAME: codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_mxfp4_q8_0_block_dot"}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong block-format fact (the mxfp4 weight stride must be 17, sizeof
// block_mxfp4 = 1 E8M0 byte + 16 packed FP4 nibbles).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 17}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight quant byte offset (the FP4 nibbles must be at +1, after the
// single E8M0 exponent byte -- NOT iq4_nl's +2, which would read into the nibbles).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_wrong_weight_quant_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_wrong_weight_quant_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_quant_byte_offset == 1}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong scale model (must be the E8M0 half-shared-exponent model; the full
// 2^(e-127) form -- the task prose's loose wording -- would double every result and
// is fail-closed, I7).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "e8m0-half-shared-exponent-per-block"}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-full-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a codebook with the wrong number of entries (must be EXACTLY 16, one per
// FP4 nibble index [0,15]). A 15-entry table cannot index the nibbles -> fail-closed.
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_wrong_codebook_size {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_wrong_codebook_size", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires codebook to carry exactly 16 int8 entries}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a non-m1 integer-core anchor (the 16-entry codebook gather needs the
// broadcast table register's VLMAX >= 16, which mf4 cannot provide at VLEN=128).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_non_m1_anchor {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_non_m1_anchor", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts integer_core_lmul "m1"}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "mf4"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type (the *s destination must be float *).
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Accept the optional multi_block_factor = 4 + strip_elision = "elided" shape (the
// loop-free half-block cover at the codebook m1 anchor, correct at VLEN >= 128; the
// autotuner gates it on Zvl128b). integer_core_lmul defaults to m1 for mxfp4.
// CHECK-LABEL: tcrv.exec.kernel @mxfp4_q8_0_block_dot_accepts_mb4_elided
module {
  tcrv.exec.kernel @mxfp4_q8_0_block_dot_accepts_mb4_elided {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "mxfp4_q8_0_block_dot_accepts_mb4_elided", status = "selected-lowering-boundary"} {
        // CHECK: strip_elision = "elided"
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "m1", multi_block_factor = 4 : i64, strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
