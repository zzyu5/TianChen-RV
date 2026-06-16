// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.q5_0_q8_0_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_q5_0_q8_0 block dot-product STRUCTURE first-class in the typed RVV
// body -- the Family-A 5-bit sibling of the Q4_0 op. It is the AoS QK=32 block
// loop, the per-block dual fp16 scale d_x.d_y, the 5-BIT weight reconstruction (a
// 4-bit nibble PLUS a per-element 5th high bit from the 32-bit qh field, with the
// offset-binary `-16` bias), the fp32 scalar accumulation, and the *s store. It
// consumes the q5_0 weight base (vx), the q8_0 activation base (vy), the fp32
// output (s), the runtime element count (n), and the active vl token, and carries
// the block-format structural facts (the strides, the WEIGHT/ACTIVATION quant byte
// offsets, the q8 high-half offset, the qh field byte offset) as typed attrs (I4
// mirror). The verifier is fail-closed (I7) on a wrong kind / scale model /
// block-format fact / operand C type / shape knob.

// CHECK-LABEL: tcrv.exec.kernel @q5_0_q8_0_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.q5_0_q8_0_block_dot
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_q5_0_q8_0_block_dot"}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong block-format fact (the q5_0 weight stride must be 22 sizeof
// block_q5_0, NOT q4_0's 18 -- q5_0 carries the 4-byte qh field).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_weight_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 22}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong WEIGHT quant byte offset (the q5_0 nibbles follow the inline fp16
// scale AND the 4-byte qh field -> +6, NOT q4_0's +2).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_quant_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_quant_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires quant_byte_offset == 6}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong ACTIVATION quant byte offset (the q8_0 int8 quants follow the
// inline fp16 scale -> +2, DISTINCT from the weight's +6 -- the q8_0 block has no
// qh field; reusing the weight's +6 silently reads the wrong activation bytes).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_act_quant_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_act_quant_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires activation_quant_byte_offset == 2}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 6 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong qh field byte offset (the block_q5_0 qh high-bit field follows the
// inline delta d -> +2).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_qh_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_qh_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_qh_byte_offset == 2}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 6 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong scale model (must be the dual-fp16 d_x.d_y model -- a plus-min
// model is fail-closed, I7; q5_0 has a single per-block scale, not q4_1's min).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "dual-fp16-per-block-d_x.d_y"}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type (the *s destination must be float *).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Accept the optional integer_core_lmul = "m1" + multi_block_factor = 4 +
// strip_elision = "elided" shape (the loop-free half-block cover at the m1 anchor,
// correct at VLEN >= 128; the autotuner gates it on Zvl128b).
// CHECK-LABEL: tcrv.exec.kernel @q5_0_q8_0_block_dot_accepts_mb4_elided
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_accepts_mb4_elided {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_accepts_mb4_elided", status = "selected-lowering-boundary"} {
        // CHECK: strip_elision = "elided"
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64, integer_core_lmul = "m1", multi_block_factor = 4 : i64, strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject strip_elision = "elided" without the m1 anchor (mf4's vsetvl_e32m1 VLMAX
// would drop 12 of 16 nibble bytes; the silently-wrong combo is fail-closed, I7).
module {
  tcrv.exec.kernel @q5_0_q8_0_block_dot_rejects_elided_non_m1 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q5_0_q8_0_block_dot_rejects_elided_non_m1", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{strip_elision "elided" requires integer_core_lmul "m1"}}
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64, integer_core_lmul = "mf4", strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
