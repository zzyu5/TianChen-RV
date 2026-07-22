// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s

// Formula-construction regression guard for the ternary GEMM leaf. Ternary has no
// output-tiling choice, but it does have the same two realized loop nests as every
// repack GEMM. The schedule formula therefore constructs loop_order independently:
// 1056 < 1168 selects the row_outer analytic prior and writes it into the final op.
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: loop_order = "row_outer"

module {
  weft.exec.kernel @ggml_repack_gemm_tq2_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT ternary PREFILL request: PLAIN block_tq2_0 (stride 66) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the base ternary tq2_0 WHAT,
        // m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "tq2_0", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
