// q5_1 honest residual-miss sibling. Historical board data is not a qualified
// current winner token, so VLEN256 decode reaches an honest residual miss.
//
// VLEN256 (rv64gcv_zvl256b => 256): no legal q5_1 block-dot path; fail closed.
// RUN: not weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b 2>&1 | FileCheck %s --check-prefix=VLEN256
//
// VLEN128 (rv64gcv => 128): UNCHANGED -- repack via the capability/regime rule
// (half_lanes 8), the q4_0-vlen128 audit token (zero rvv drift).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128

module {
  weft.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q5_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min-five-bit", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 24 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (VLEN256 tier) honest residual miss fails closed because q5_1 has no legal
// block-dot decline path.
// VLEN256: q5_0 / q5_1 / q8_0 quant_contraction requires a repack-affording capability

// (VLEN128 tier) UNCHANGED capability/regime repack (zero rvv drift).
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 8 : i64
