// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Track B q4_K BRICK 2: the weft_rvv.q4_k_scale_min_bit_dance op makes the
// q4_K/q5_K Region-B 6-bit scale/min bit-dance (the get_scale_min_k4 / utmp/kmask
// cross-byte decode into the 16 [scales,mins] bytes) first-class in the typed RVV
// body (the bounded auto-construction vocabulary for the q4_K integer core,
// sibling to BRICK 1's weft_rvv.q4_k_nibble_unpack). It consumes ONE q4_K weight
// base pointer (vx, the AoS block_q4_K byte array) + the active vl token, carries
// only the super-block Region-B format facts as typed attrs (I4 mirror), and
// produces an i32 m1 completion token (the bit-dance writes utmp as a side
// effect). NO plain 4-bit nibble unpack, NO per-sub-block dot, NO MIN term, NO
// fp32 fold -- those are BRICK 1 / deferred bricks. The verifier is fail-closed
// (I7) on a wrong kind / block-format fact / weight C type / unexpected attr.

// CHECK-LABEL: weft.exec.kernel @q4_k_scale_min_bit_dance_accepts_ggml_abi
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_accepts_ggml_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // CHECK: weft_rvv.q4_k_scale_min_bit_dance
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong operation kind (fail-closed, I7).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{currently supports only kind "q4_k_scale_min_bit_dance"}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "block_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong super-block size (q4_K is QK_K = 256, NOT a flat-family QK = 32).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_wrong_qk {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires qk == 256 (QK_K)}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 32 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong sub-block size (q4_K is 32-element sub-blocks).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_wrong_sub_block {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires sub_block == 32}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight stride (block_q4_K is 144 bytes: d+dmin+scales[12]+qs[128]).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_wrong_weight_stride {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires weight_block_stride in}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 210 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong scales byte offset (q4_K scales follow d+dmin -> +4).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_wrong_scales_offset {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires weight_scales_byte_offset == 4}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a wrong weight base operand C type. The Region-B weight base addresses
// the AoS block_q4_K byte array as const uint8_t * -- a float * is fail-closed.
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_wrong_weight_ctype {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{requires the weight base operand to bind a runtime ABI value of C type 'const uint8_t *'}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject an unexpected attribute (the bounded surface is exactly kind + the four
// Region-B format facts). Fail-closed (I7).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_unexpected_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{only accepts the bounded scale-min-bit-dance attributes}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// Reject a FORBIDDEN dataflow-parameter attr leaking as a local op metadata
// (element_count / SEW / LMUL / policy belong on setvl/with_vl, NOT on the typed
// dataflow op). This exercises the isForbiddenDataflowParameterAttr guard branch
// -- distinct from the "only accepts the bounded ..." unexpected-attr branch
// above -- mirroring the BRICK 1 nibble-unpack sibling's fail-closed dataflow
// discipline (I7).
module {
  weft.exec.kernel @q4_k_scale_min_bit_dance_rejects_forbidden_dataflow_attr {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // expected-error @+1 {{does not accept attribute}}
        %u = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64, element_count = 256 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
