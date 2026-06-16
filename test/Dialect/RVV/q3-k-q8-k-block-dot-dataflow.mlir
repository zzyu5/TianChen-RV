// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.q3_k_q8_k_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_q3_K_q8_K (q3_K coverage, the LAST common K-quant) first-class in
// the typed RVV body. q3_K COMPOSES q2_K's 2-bit qs unpack + the SUBTRACTIVE
// hmask high-bit injection (q5_K's fixed 32-byte plane but -4 when the bit is
// UNSET -> SIGNED [-4,3]) + the q3_K-OWN SIGNED 6-bit scale dance (scales[j]-32,
// masks 0x03030303 / 0x0f0f0f0f) + q6_K's NO-min deferred d.Sum(aux32) fold (q3_K
// is symmetric -- NO min term, NO dmin). Its single output is the fp32 *s (a
// float *). It consumes the q3_K weight base (vx), the q8_K activation base (vy),
// the float* output (s), the runtime element count (n), and the active vl token,
// and carries the super-block-format structural facts as typed attrs (I4 mirror)
// INCLUDING the q3_K-specific hmask @0 plane offset (qs @32, scales @96, d @108,
// stride 110). The verifier is fail-closed (I7) on a wrong kind / scale model /
// block-format fact / operand C type.

// CHECK-LABEL: tcrv.exec.kernel @q3_k_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @q3_k_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.q3_k_q8_k_block_dot
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_q3_k_q8_k_block_dot"}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong hmask byte offset (q3_K's 32-byte high-bit plane LEADS the
// block -> +0). The hmask plane is the q3_K-distinguishing field; pin it.
// Fail-closed (I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_wrong_hmask_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_wrong_hmask_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_hmask_byte_offset == 0}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 32 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong scales byte offset (q3_K scales follow hmask[32]+qs[64] -> +96,
// NOT q4_K's +4 or q2_K's +0). Fail-closed (I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_wrong_scales_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_wrong_scales_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_scales_byte_offset == 96}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 4 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight block stride (sizeof block_q3_K == 110). Fail-closed (I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_wrong_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_wrong_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 110}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject the q4_K/q5_K/q2_K min-bearing scale model (q3_K is SYMMETRIC -- NO min
// term, NO dmin). Fail-closed (I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_min_scale_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_min_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold"}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type. q3_K's *s destination is a float * (the
// fp32 dot-product output) -- fail-closed (I7).
module {
  tcrv.exec.kernel @q3_k_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q3_k_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.q3_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q3_k_q8_k_block_dot", scale_model = "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, weight_d_byte_offset = 108 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
