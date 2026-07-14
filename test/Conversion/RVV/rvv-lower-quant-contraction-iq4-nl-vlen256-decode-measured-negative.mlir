// OPTION-2 STAGE B -- [G8 六.3] the iq4_nl NEGATIVE side of the PER-FORMAT MEASURED
// VLEN256-DECODE cell (the decisive contrast to q5_0/q5_1). iq4_nl measured repack
// NEGATIVE 0.248x at the k1 VLEN256 decode cell (codebook-gather-bound; casefile
// experiments/active/g8-stage3-attack/k1-gevm-sweep, commit ac5ea76f), so the
// board-seeded kRepackVlen256DecodeMeasurements registry drives
// vlen256DecodeRepackBeneficial=false and the selector DECLINES repack at the SAME cell
// that q5_0/q5_1 WIN. Because the codebook family has NO block-dot decline path (the
// block-dot identity lowering is q4_0-nibble-only), the DECLINE then fails closed (I7).
// This is the SAME behavior as before the per-format fix (the old blanket rule ALSO
// declined iq4_nl at VLEN256) -- so this fixture is the NO-REGRESSION guard proving the
// per-format registry did NOT accidentally flip iq4_nl to repack.
//
// VLEN256 (rv64gcv_zvl256b => 256): iq4_nl decode -> DECLINE (measured-negative) -> no
// block-dot fallback -> capability error (asserted via --verify-diagnostics).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --verify-diagnostics

module {
  weft.exec.kernel @ggml_repack_gemv_iq4_nl_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires a repack-affording capability}}
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq4_nl", scale_model = "flat.fp16-single-scale-codebook-nomin", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
