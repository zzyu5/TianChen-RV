// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction | FileCheck %s

// OPTION-2 STAGE B -- the q4_0 PREFILL cell of the IN-COMPILER contraction-path
// SELECTION. (Stage A made this cell a fail-closed ERROR STUB; stage B makes it
// reachable-and-handled: the selector's fact 3 keeps Repack for ANY prefill
// regime, M-amortized, INDEPENDENT of VLEN -- so q4_0 prefill selects Repack even
// at the default -march "" (VLEN 0).)
//
// Per the Option-(i) crux, Repack-SELECTED does NOT materialize a repack op (the
// plain->x16 weight materialization is stage C). It emits the byte-identical
// tcrv_rvv.q4_0_q8_0_block_dot body + the audit attrs recording that REPACK was
// selected but its materialization is DEFERRED to stage C. The emitted C is
// byte-identical to the decode-realized block-dot; only the audit token differs.
//
// LATENT-MISPICK HONESTY: fact 3 keeps Repack for ANY prefill, so q4_0 @
// K1-VLEN256 *prefill* is auto-kept though only the *decode* cell was measured.
// This is a known latent mispick recorded in the design, not a validated cell.

module {
  tcrv.exec.kernel @quant_contraction_prefill_select {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "quant_contraction_prefill_select", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract op is lowered; NO repack op is materialized (Option (i)); the
// prefill cell audits REPACK selected, materialization deferred to stage C.
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// CHECK: tcrv_rvv.q4_0_q8_0_block_dot
// CHECK-SAME: tcrv_rvv.contraction_algorithm = "repack"
// CHECK-SAME: tcrv_rvv.path_materialization = "deferred-stage-c"
// CHECK-SAME: tcrv_rvv.path_selection_reason = "repack-kept-q4_0-prefill"
