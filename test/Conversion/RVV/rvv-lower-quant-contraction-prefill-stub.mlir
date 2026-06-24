// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-lower-quant-contraction --verify-diagnostics

// OPTION-2 STAGE A -- the --tcrv-rvv-lower-quant-contraction PASS fail-closed
// stub coverage (the verifier-vs-pass split). The verifier ACCEPTS m_regime in
// {decode, prefill} and any supported quant, but the stage-A pass wires ONLY the
// q4_0-decode block-dot identity branch. Every other (quant, m_regime) cell is
// the stage-C/repack lowering target and is a deliberate ERROR STUB here: a
// misrouted op must FAIL fail-closed (I7), NOT silently fall through to
// block-dot. This fixture exercises that PASS-level error path (the dataflow
// verifier test only covers the verifier; this proves the pass itself does not
// silently miscompile a non-decode request).

// A verifier-VALID q4_0 + prefill request (the verifier accepts prefill) that
// the PASS must reject -- it is the repack/GEMM cell, not yet wired.
module {
  tcrv.exec.kernel @quant_contraction_prefill_pass_stub {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "quant_contraction_prefill_pass_stub", status = "selected-lowering-boundary"} {
        // expected-error @+1 {{stage-A quant_contraction lowering wires ONLY the q4_0 decode block-dot identity branch}}
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
