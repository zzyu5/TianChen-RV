// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The weft_rvv.q1_0_q8_0_binary_sign_core op makes the bounded ggml
// ggml_vec_dot_q1_0_q8_0 block dot-product STRUCTURE first-class in the typed RVV
// body -- the BINARY ({-1,+1}) class, one of the last three uncommon ggml dot
// kernels. The q1_0 weight is a 1-bit-per-element sign (a SET bit -> +q8, a CLEAR
// bit -> -q8; the q8 value is the magnitude). block_q1_0 = { ggml_half d;
// uint8_t qs[16] } (stride 18, 128 element bits at +2), and ONE 128-element
// super-block spans FOUR block_q8_0 activation blocks (sub-block k -> q8 block
// 4*ib+k, its 32 bits at weight bytes +2 + k*4 .. +3). It consumes the q1_0 weight
// base (vx), the q8_0 activation base (vy), the runtime element
// count (n), and the active vl token, and carries the super-block-format structural
// facts (qk, the strides, the per-super-block q8-block span, the two quant byte
// offsets) as typed attrs (I4 mirror). The verifier is fail-closed (I7) on a wrong
// kind / scale model / super-block-format fact / out-of-set anchor / an anchor whose
// i8 VLMAX does not span the 32-element sub-block at the minimum_vlen / operand C type.

// CHECK-LABEL: weft.exec.kernel @q1_0_q8_0_binary_sign_core_accepts_ggml_abi
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_accepts_ggml_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        // CHECK: weft_rvv.q1_0_q8_0_binary_sign_core
        // CHECK-SAME: scale_model = "binary-sign-per-bit"
        // The VLEN128-legal anchor (m2 spans the 32-element sub-block: e8m2 VLMAX 32).
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m2", minimum_vlen = 128 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Accept the m1 anchor at minimum_vlen 256 (where e8m1 VLMAX 32 spans the sub-block).
// CHECK-LABEL: weft.exec.kernel @q1_0_q8_0_binary_sign_core_accepts_m1_at_vlen256
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_accepts_m1_at_vlen256 {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_accepts_m1_at_vlen256", status = "selected-lowering-boundary"} {
        // CHECK: weft_rvv.q1_0_q8_0_binary_sign_core
        // CHECK-SAME: integer_core_lmul = "m1"
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m1", minimum_vlen = 256 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "ggml_q1_0_q8_0_binary_sign_core"}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "block_dot", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong scale model (must be the binary-sign-per-bit model; a codebook or
// nibble decode misroute is fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_wrong_scale_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_wrong_scale_model", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires scale_model "binary-sign-per-bit"}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "offset-binary-per-nibble", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong qk (must be 128 = QK1_0; fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_wrong_qk {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_wrong_qk", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires qk == 128 (QK1_0)}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight_block_stride (must be 18 = sizeof block_q1_0; fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_wrong_weight_stride {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride == 18}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong activation_blocks_per_weight (must be 4; fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_wrong_q8_span {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_wrong_q8_span", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires activation_blocks_per_weight == 4}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 2 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject an out-of-set integer-core anchor (only the whole-LMUL {m1, m2} anchors are
// legal for the 32-lane binary sign decode; fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_out_of_set_anchor {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_out_of_set_anchor", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts integer_core_lmul "m1" or "m2"}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "mf4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject the m1 anchor at minimum_vlen 128 (the SILENT-WRONG VLEN guard, I7): e8m1's
// VLMAX is 16 at VLEN128, so a single vsetvl_e8m1(32) cover would process only 16 of
// the 32 sub-block lanes and DROP the rest. The verifier recomputes this from the
// SAME getRVVStripVLMAXElements formula the gearbox selects with, so m1 is admitted
// ONLY at minimum_vlen >= 256.
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_m1_at_vlen128 {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_m1_at_vlen128", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires an integer_core_lmul whose i8 strip VLMAX spans the 32-element q8 sub-block at the guaranteed minimum_vlen (128)}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m1", minimum_vlen = 128 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight operand C type (must be const uint8_t *; fail-closed, I7).
module {
  weft.exec.kernel @q1_0_q8_0_binary_sign_core_rejects_wrong_weight_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q1_0_q8_0_binary_sign_core_rejects_wrong_weight_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the weight base operand to bind a runtime ABI value of C type 'const uint8_t *'}}
        %sumi = weft_rvv.q1_0_q8_0_binary_sign_core %vx, %vy, %n, %vl {kind = "ggml_q1_0_q8_0_binary_sign_core", scale_model = "binary-sign-per-bit", qk = 128 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, activation_blocks_per_weight = 4 : i64, weight_quant_byte_offset = 2 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m2", minimum_vlen = 128 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}
