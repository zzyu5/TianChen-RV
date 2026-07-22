// OPTION-2 STAGE B/C1 -- [G8 六.3] the q5_1 sibling of the PER-FORMAT MEASURED
// VLEN256-DECODE cell (see rvv-lower-quant-contraction-vlen256-decode-per-format-
// measured.mlir for the q5_0 headline + the full rationale). q5_1 measured BENEFICIAL
// 1.306x at the k1 VLEN256 decode cell (casefile
// experiments/active/g8-stage3-attack/k1-gevm-sweep, commit ac5ea76f), so the
// board-seeded kRepackVlen256DecodeMeasurements registry drives the fact
// vlen256DecodeRepackBeneficial=true and the pure (format-blind) selector SELECTS repack
// at the SAME cell that DECLINES for q4_0/iq4_nl. NO perf/e2e claim -- lit-emitted, NOT run.
//
// VLEN256 (rv64gcv_zvl256b => 256): q5_1 decode -> REPACK via the measured-BENEFICIAL
// fact -> typed weft_rvv.typed_repack_gemv_loop_body (half_lanes 16).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=VLEN256
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

// (VLEN256 tier) measured-BENEFICIAL fact SELECTS repack; the abstract op is gone, no
// block-dot is emitted, the typed repack-GEVM region is realized (half_lanes 16, x16).
// VLEN256-NOT: weft_rvv.quant_contraction
// VLEN256-NOT: weft_rvv.q5_1_q8_1_block_dot
// VLEN256: weft_rvv.typed_repack_gemv_loop_body
// VLEN256-SAME: half_lanes = 16 : i64
// VLEN256-SAME: weft_rvv.weight_layout_contract = "x16"

// (VLEN128 tier) UNCHANGED capability/regime repack (zero rvv drift).
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 8 : i64
