// The I7 fail-closed guard for the CODEBOOK class (Win-A brick #2b收尾), MXFP4
// sibling, END-TO-END: an attr-less ggml MXFP4 x Q8_0 codebook op materialize-
// scheduled at a SUB-128 tier (rv64gc_zve32x, VLEN < 128) gets NO legal anchor
// stamped (the codebook class is Zvl128b-gated -- its 16-entry gather needs a strip
// VLMAX >= 16, unreachable below VLEN=128), so it reaches the lowering boundary
// UN-scheduled. The emitter MUST refuse it fail-closed (NOT default to m1 and emit a
// broken gather that silently reads 0 for a high nibble index). This is the EXACT
// pipeline that previously lowered to a live vsetvl_e8m1/vrgather_vv_i8m1 at EXIT 0.
//
// The guard lives at the lowering boundary, not the verifier (SAME as the iq4_nl
// sibling): an attr-less op with no minimum_vlen is the LEGAL pre-schedule input the
// materialize-schedule pass must verify, and the sub-128 pass run leaves it attr-less
// with NO minimum_vlen -- verifier-indistinguishable from the legal input. The
// emitter CAN refuse it: every VLEN>=128 path stamps a legal m1/mf2 first.
//
// RUN: not tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gc_zve32x --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s

// The conversion fails (the codebook op was NOT legalized) and -- the load-bearing
// part -- NO broken sub-128 gather is emitted: NO emitc.func, NO e8m1 strip vsetvl,
// NO i8m1 codebook gather. The emitter refused fail-closed (the reason string is on
// the notifyMatchFailure remark; the user-visible signal is the failed legalization
// + the ABSENCE of the broken code, same shape as the other negative emitc lit).
// CHECK: error
// CHECK-NOT: emitc.func
// CHECK-NOT: __riscv_vsetvl_e8m1
// CHECK-NOT: __riscv_vrgather_vv_i8m1

module {
  tcrv.exec.kernel @ggml_vec_dot_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
