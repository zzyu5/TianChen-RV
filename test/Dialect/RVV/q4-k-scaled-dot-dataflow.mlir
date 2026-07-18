// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Track B q4_K BRICK 3: the weft_rvv.q4_k_scaled_dot op makes the q4_K/q5_K
// Region-C per-sub-block uint6-scaled i32 dot into the 8-lane aux32 + the integer
// fold-back first-class in the typed RVV body (the bounded auto-construction
// vocabulary for the q4_K integer core, sibling to BRICK 1's
// weft_rvv.q4_k_nibble_unpack and BRICK 2's weft_rvv.q4_k_scale_min_bit_dance). It
// consumes THREE base pointers -- the BRICK 1 unpacked aux8 (const int8_t *), the
// BRICK 2 decoded scales (const uint8_t *), the q8_K activation (const uint8_t *)
// -- plus the active vl token, carries the super-block Region-C format facts + an
// OPTIONAL integer_core_lmul resource anchor as typed attrs (I4 mirror), and
// produces an i32 m1 completion token (the dot writes aux32 as a side effect). NO
// nibble unpack, NO bit-dance, NO MIN term, NO fp32 fold -- those are BRICK 1 /
// BRICK 2 / deferred bricks. The verifier is fail-closed (I7) on a wrong kind /
// block-format fact / base C type / unexpected attr / illegal integer_core_lmul.

// CHECK-LABEL: weft.exec.kernel @q4_k_scaled_dot_accepts_default
module {
  weft.exec.kernel @q4_k_scaled_dot_accepts_default {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_accepts_default", status = "selected-lowering-boundary"} {
        // CHECK: weft_rvv.q4_k_scaled_dot
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Accept the WIDE m2 integer_core_lmul anchor (the legal resource knob the q4_K
// capability flip rides on -- the fold-back). It must roundtrip unchanged.
// CHECK-LABEL: weft.exec.kernel @q4_k_scaled_dot_accepts_m2_anchor
module {
  weft.exec.kernel @q4_k_scaled_dot_accepts_m2_anchor {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_accepts_m2_anchor", status = "selected-lowering-boundary"} {
        // CHECK: integer_core_lmul = "m2"
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "m2", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Accept the ISSUE-109 "vwredsum" integer_core_lmul anchor (the aux8-free
// per-sub-block independent vwredsum.vs reduce lever, q4_K non-qh only). It must
// roundtrip unchanged.
// CHECK-LABEL: weft.exec.kernel @q4_k_scaled_dot_accepts_vwredsum_anchor
module {
  weft.exec.kernel @q4_k_scaled_dot_accepts_vwredsum_anchor {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_accepts_vwredsum_anchor", status = "selected-lowering-boundary"} {
        // CHECK: integer_core_lmul = "vwredsum"
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "vwredsum", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Accept the ISSUE-109 "minterm-vec" integer_core_lmul anchor (the aux8-free
// vwredsum block-dot PLUS the wide-LMUL vectorized MIN-term reduction, q4_K
// non-qh only). It must roundtrip unchanged.
// CHECK-LABEL: weft.exec.kernel @q4_k_scaled_dot_accepts_minterm_vec_anchor
module {
  weft.exec.kernel @q4_k_scaled_dot_accepts_minterm_vec_anchor {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_accepts_minterm_vec_anchor", status = "selected-lowering-boundary"} {
        // CHECK: integer_core_lmul = "minterm-vec"
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "minterm-vec", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "q4_k_scaled_dot"}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "block_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong super-block size (q4_K is QK_K = 256, NOT a flat-family QK = 32).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_qk {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_qk", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires qk == 256 (QK_K)}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 32 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block size (q4_K is 32-element sub-blocks).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_sub_block {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_sub_block", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires sub_block == 32}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight stride (block_q4_K is 144 bytes: d+dmin+scales[12]+qs[128]).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_weight_stride {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_weight_stride", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires weight_block_stride in}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 210 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong aux8 base operand C type. The BRICK 1 unpacked aux8 addresses the
// int8_t aux8[256] scratch as const int8_t * -- a const float * is fail-closed.
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_aux8_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_aux8_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the aux8 base operand to bind a runtime ABI value of C type 'const int8_t *'}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong scales base operand C type. The BRICK 2 decoded scales address
// the 6-bit scales as const uint8_t * -- a const float * is fail-closed.
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_scales_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_scales_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the scales base operand to bind a runtime ABI value of C type 'const uint8_t *'}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong q8 base operand C type. The q8_K activation addresses the block
// byte array as const uint8_t * -- a const float * is fail-closed.
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_wrong_q8_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_wrong_q8_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the q8 base operand to bind a runtime ABI value of C type 'const uint8_t *'}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject an illegal integer_core_lmul value. The legal set is {"mf2","m1","m2",
// "fused","vwredsum","minterm-vec"} (the base LMUL of the i8 -> i16 -> i32
// integer-MAC chain, plus the ISSUE-109 "fused" register-resident aux8-free
// anchor, the "vwredsum" per-sub-block independent-reduce anchor, and the
// "minterm-vec" vectorized-min-term anchor; "m4" would need an illegal i32m16
// product). Fail-closed (I7).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_illegal_lmul {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_illegal_lmul", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires integer_core_lmul in {"mf2", "m1", "m2", "fused", "vwredsum", "minterm-vec"}}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "m4", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject an unexpected attribute (the bounded surface is exactly kind + the three
// Region-C format facts + the optional integer_core_lmul). Fail-closed (I7).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_unexpected_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_unexpected_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts the bounded scaled-dot attributes}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_qs_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a FORBIDDEN dataflow-parameter attr leaking as local op metadata
// (element_count / SEW / LMUL / policy belong on setvl/with_vl, NOT on the typed
// dataflow op). This exercises the isForbiddenDataflowParameterAttr guard branch
// -- distinct from the "only accepts the bounded ..." unexpected-attr branch above
// -- mirroring the BRICK 1 / BRICK 2 sibling's fail-closed dataflow discipline (I7).
module {
  weft.exec.kernel @q4_k_scaled_dot_rejects_forbidden_dataflow_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_rejects_forbidden_dataflow_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{does not accept attribute}}
        %d = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, element_count = 256 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
