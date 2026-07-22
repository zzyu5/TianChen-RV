// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Track B q4_K BRICK 7: the weft_rvv.q4_k_horizontal_fold op makes the q4_K/q5_K
// block dot's POST-LOOP horizontal fold (vse32 the 8-lane fp32 sums into a sums8[8]
// scratch, then the SEQUENTIAL `for (l=0..7) sumf += sums8[l]`) first-class in the
// typed RVV body (the bounded auto-construction vocabulary for the q4_K block dot,
// sibling to BRICK 1's weft_rvv.q4_k_nibble_unpack, BRICK 2's
// weft_rvv.q4_k_scale_min_bit_dance, BRICK 3's weft_rvv.q4_k_scaled_dot, BRICK 4's
// weft_rvv.q4_k_min_term, and BRICK 6's weft_rvv.q4_k_sums_fold_scale_d). It
// consumes ONE base pointer -- the 8-lane fp32 sums source (const float *,
// vle32-loaded into a vfloat32m2_t) -- plus the active vl token, carries the
// super-block format facts as typed attrs (I4 mirror), and produces an i32 m1
// completion token (the fold mutates the scalar sumf + a local sink as a side
// effect). NO nibble unpack, NO bit-dance, NO scaled dot, NO MIN term, NO positive
// fold -- those are the sibling bricks. The verifier is fail-closed (I7) on a wrong
// kind / block-format fact / base C type / unexpected attr.

// CHECK-LABEL: weft.exec.kernel @q4_k_horizontal_fold_accepts_default
module {
  weft.exec.kernel @q4_k_horizontal_fold_accepts_default {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // CHECK: weft_rvv.q4_k_horizontal_fold
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{currently supports only kind "q4_k_horizontal_fold"}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "block_dot", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong super-block size (q4_K is QK_K = 256, NOT a flat-family QK = 32).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_wrong_qk {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires qk == 256 (QK_K)}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 32 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block size (q4_K is 32-element sub-blocks).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_wrong_sub_block {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires sub_block == 32}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 16 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block count (q4_K is QK_K / 32 = 8 sub-blocks).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_wrong_num_sub_blocks {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires num_sub_blocks == 8}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 4 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong fp32 lane count (the canonical q4_K/q5_K sums accumulator is 8
// lanes -- num_lanes == 8).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_wrong_num_lanes {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires num_lanes == 8}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong sums base operand C type. The 8-lane fp32 sums source is addressed
// as const float * (vle32-loaded into a vfloat32m2_t) -- a const uint8_t * is
// fail-closed.
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_wrong_sums_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires the sums base operand to bind a runtime ABI value of C type 'const float *'}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject an unexpected attribute (the bounded surface is exactly kind + the four
// format facts). Fail-closed (I7).
module {
  weft.exec.kernel @q4_k_horizontal_fold_rejects_unexpected_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{only accepts the bounded horizontal-fold attributes}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64, weight_d_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
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
  weft.exec.kernel @q4_k_horizontal_fold_rejects_forbidden_dataflow_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = weft_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{does not accept attribute}}
        %fd = weft_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64, element_count = 256 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
