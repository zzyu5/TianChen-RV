// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.q2_k_q8_k_block_dot op makes the COMPLETE ggml
// ggml_vec_dot_q2_K_q8_K (q2_K coverage) first-class in the typed RVV body. q2_K
// is the 2-bit modern K-quant: it REUSES the super-block scaffolding (the
// QK_K=256 AoS super-block loop, the q8_K activation handling, the
// `(float)*(const _Float16 *)` fp16 read seam, the structured `*s` store) and the
// scale+min structure of q4_K, but is SIMPLER in three ways: (1) the weights are
// 2-bit (4 per qs byte), unpacked by `(qs >> shift) & 3`; (2) the per-sub-block
// scales/mins are SIMPLE 4-bit nibbles of the 16 direct `scales[16]` bytes
// (`sc[j] & 0xF` scale, `sc[j] >> 4` min -- NO 6-bit bit-dance); (3) the positive
// fold is SCALAR -- a single per-super-block int isum, folded `sumf += dall*isum
// - dmin*summs` as one scalar fp32 statement (NO 8-lane sums vector). Its single
// output is the fp32 *s (a float *). It consumes the q2_K weight base (vx), the
// q8_K activation base (vy), the float* output (s), the runtime element count
// (n), and the active vl token, and carries the super-block-format structural
// facts as typed attrs (I4 mirror): scales @0, qs @16, d @80, dmin @82, stride
// 84, 16 sub-blocks of 16. The verifier is fail-closed (I7) on a wrong kind /
// scale model / block-format fact / operand C type.

// CHECK-LABEL: tcrv.exec.kernel @q2_k_block_dot_accepts_ggml_abi
module {
  tcrv.exec.kernel @q2_k_block_dot_accepts_ggml_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.q2_k_q8_k_block_dot
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_q2_k_q8_k_block_dot"}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight scales byte offset (q2_K scales[16] LEAD block_q2_K ->
// +0, distinct from q4_K's +4). Fail-closed (I7).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_wrong_scales_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_wrong_scales_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_scales_byte_offset == 0}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub_block (q2_K = 16 sub-blocks of 16, NOT q4_K's 32). The
// 16-element sub-block boundary is q2_K's distinguishing scale granularity; pin
// it. Fail-closed (I7).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_wrong_sub_block {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_wrong_sub_block", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires sub_block == 16}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight block stride (sizeof block_q2_K == 84, NOT q4_K's 144).
// Fail-closed (I7).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_wrong_stride {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_wrong_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 84}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong bsums byte offset (q8_K bsums follow d+qs[256] -> +260).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_wrong_bsums_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_wrong_bsums_offset", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires activation_bsums_byte_offset == 260}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong output operand C type. q2_K's *s destination is a float * (the
// fp32 dot-product output) -- fail-closed (I7).
module {
  tcrv.exec.kernel @q2_k_block_dot_rejects_wrong_output_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q2_k_block_dot_rejects_wrong_output_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the output operand to bind a runtime ABI value of C type 'float *'}}
        %dot = tcrv_rvv.q2_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q2_k_q8_k_block_dot", scale_model = "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, weight_d_byte_offset = 80 : i64, weight_dmin_byte_offset = 82 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
