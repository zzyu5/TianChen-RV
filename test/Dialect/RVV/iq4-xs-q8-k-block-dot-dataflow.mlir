// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.iq4_xs_q8_k_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_iq4_xs_q8_K super-block dot-product STRUCTURE first-class in the typed
// RVV body -- the CODEBOOK class's SUPER-BLOCK rung (iq4_xs is the super-block variant
// of iq4_nl). It is the AoS QK_K=256 super-block loop, the per-super-block d4d8 scale
// (fp16 weight d x fp32 q8_K d), the per-sub-block SIGNED 6-bit scale extracted from
// scales_l[4]+scales_h biased -32, the NON-LINEAR codebook decode (each 4-bit nibble
// indexes the SAME 16-entry kvalues_iq4nl[16] int8 table iq4_nl uses), the FLOAT-domain
// per-sub-block fold (NO min, symmetric), and the *s store. It consumes the iq4_xs
// weight base (vx), the q8_K activation base (vy), the fp32 output (s), the runtime
// element count (n), and the active vl token, and carries the super-block-format
// structural facts (strides, byte offsets) PLUS the 16-entry codebook as typed attrs
// (I4 mirror). The verifier is fail-closed (I7) on a wrong kind / scale model /
// block-format fact / codebook size / operand C type.

// CHECK-LABEL: tcrv.exec.kernel @iq4_xs_q8_k_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.iq4_xs_q8_k_block_dot
        // CHECK-SAME: codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_iq4_xs_q8_k_block_dot"}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong block-format fact (the iq4_xs weight stride must be 136, sizeof
// block_iq4_xs).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 136}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong qs byte offset (must be 8, after d[2]+scales_h[2]+scales_l[4]).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_wrong_qs_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_wrong_qs_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_qs_byte_offset == 8}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a codebook with the wrong number of entries (must be EXACTLY 16, the SAME
// kvalues_iq4nl[16] iq4_nl uses; the codebook is the load-bearing structural fact of
// the codebook class). A 15-entry table cannot index the nibbles -> fail-closed (I7).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_wrong_codebook_size {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_wrong_codebook_size", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires codebook to carry exactly 16 int8 entries}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong scale model (must be the float-domain signed codebook scale model;
// a deferred-int-domain model -- q6_K's -- is fail-closed because iq4_xs MUST fold in
// float to match _generic's two roundings under -ffp-contract=off).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "per-sub-block-int6-signed-codebook-scale-float-domain"}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int8-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type (the *s destination must be float *).
module {
  tcrv.exec.kernel @iq4_xs_q8_k_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4xs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "iq4_xs_q8_k_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.iq4_xs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_xs_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-codebook-scale-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
