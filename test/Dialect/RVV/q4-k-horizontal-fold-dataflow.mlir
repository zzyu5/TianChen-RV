// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Track B q4_K BRICK 7: the tcrv_rvv.q4_k_horizontal_fold op makes the q4_K/q5_K
// block dot's POST-LOOP horizontal fold (vse32 the 8-lane fp32 sums into a sums8[8]
// scratch, then the SEQUENTIAL `for (l=0..7) sumf += sums8[l]`) first-class in the
// typed RVV body (the bounded auto-construction vocabulary for the q4_K block dot,
// sibling to BRICK 1's tcrv_rvv.q4_k_nibble_unpack, BRICK 2's
// tcrv_rvv.q4_k_scale_min_bit_dance, BRICK 3's tcrv_rvv.q4_k_scaled_dot, BRICK 4's
// tcrv_rvv.q4_k_min_term, and BRICK 6's tcrv_rvv.q4_k_sums_fold_scale_d). It
// consumes ONE base pointer -- the 8-lane fp32 sums source (const float *,
// vle32-loaded into a vfloat32m2_t) -- plus the active vl token, carries the
// super-block format facts as typed attrs (I4 mirror), and produces an i32 m1
// completion token (the fold mutates the scalar sumf + a local sink as a side
// effect). NO nibble unpack, NO bit-dance, NO scaled dot, NO MIN term, NO positive
// fold -- those are the sibling bricks. The verifier is fail-closed (I7) on a wrong
// kind / block-format fact / base C type / unexpected attr.

// CHECK-LABEL: tcrv.exec.kernel @q4_k_horizontal_fold_accepts_default
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_accepts_default {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_accepts_default", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.q4_k_horizontal_fold
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{currently supports only kind "q4_k_horizontal_fold"}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "block_dot", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong super-block size (q4_K is QK_K = 256, NOT a flat-family QK = 32).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_wrong_qk {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_wrong_qk", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires qk == 256 (QK_K)}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 32 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block size (q4_K is 32-element sub-blocks).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_wrong_sub_block {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_wrong_sub_block", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires sub_block == 32}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 16 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block count (q4_K is QK_K / 32 = 8 sub-blocks).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_wrong_num_sub_blocks {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_wrong_num_sub_blocks", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires num_sub_blocks == 8}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 4 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong fp32 lane count (the canonical q4_K/q5_K sums accumulator is 8
// lanes -- num_lanes == 8).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_wrong_num_lanes {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_wrong_num_lanes", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires num_lanes == 8}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a wrong sums base operand C type. The 8-lane fp32 sums source is addressed
// as const float * (vle32-loaded into a vfloat32m2_t) -- a const uint8_t * is
// fail-closed.
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_wrong_sums_ctype {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_wrong_sums_ctype", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{requires the sums base operand to bind a runtime ABI value of C type 'const float *'}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject an unexpected attribute (the bounded surface is exactly kind + the four
// format facts). Fail-closed (I7).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_unexpected_attr {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_unexpected_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{only accepts the bounded horizontal-fold attributes}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64, weight_d_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// Reject a FORBIDDEN dataflow-parameter attr leaking as local op metadata
// (element_count / SEW / LMUL / policy belong on setvl/with_vl, NOT on the typed
// dataflow op). This exercises the isForbiddenDataflowParameterAttr guard branch
// -- distinct from the "only accepts the bounded ..." unexpected-attr branch above
// -- mirroring the sibling bricks' fail-closed dataflow discipline (I7).
module {
  tcrv.exec.kernel @q4_k_horizontal_fold_rejects_forbidden_dataflow_attr {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_rejects_forbidden_dataflow_attr", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{does not accept attribute}}
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64, element_count = 256 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
