// The AUTO-DISCOVERY proof for the unified schedule autotuner
// (--tcrv-rvv-materialize-schedule): ONE walk-all pass selects + stamps the
// schedule of EVERY op that implements the TunableScheduleOpInterface, with NO
// per-op pass and NO hardcoded op-type list. The pass walks via
// dyn_cast<TunableScheduleOpInterface> and looks each op's kernel key up in the
// plugin-local descriptor registry.
//
// This module carries TWO STRUCTURALLY DIFFERENT tunable ops -- a Q4_0 x Q8_0
// block-dot (the LMUL/factor/elision triple, vector_register_budget audit flavor)
// and a Q4_0 x Q8_0 FULL GEMM (a single activation_cols M knob, vreg_ceiling audit
// flavor) -- neither carrying a hand-authored shape knob. ONE invocation of the
// unified pass must stamp BOTH, each through its OWN descriptor, proving the pass
// auto-discovers any tunable op rather than enumerating a fixed set.
//
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s

module {
  // (A) A Q4_0 x Q8_0 block-dot op, attr-less -- the block-dot descriptor (the
  // LMUL/factor/elision triple) must be selected + stamped.
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }

  // (B) A Q4_0 x Q8_0 FULL GEMM op, attr-less -- the GEMM descriptor (the single
  // activation_cols M knob) must be selected + stamped, in the SAME pass.
  tcrv.exec.kernel @gemm {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @gemm attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @gemm, sew = 32 : i64, source_kernel = "gemm", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.q4_0_q8_0_gemm %vx, %vy, %by, %s, %n, %nr, %nc, %bx, %bs, %vl {kind = "ggml_q4_0_q8_0_gemm", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.runtime_abi_value, index, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// (A) The block-dot op was auto-discovered + stamped through the block-dot
// descriptor: the LMUL/factor/elision triple + the "tcrv_rvv.q4_0_schedule.*"
// provenance (vector_register_budget audit flavor).
// CHECK: tcrv_rvv.q4_0_q8_0_block_dot
// CHECK-SAME: integer_core_lmul = "m1"
// CHECK-SAME: multi_block_factor = 4 : i64
// CHECK-SAME: strip_elision = "elided"
// CHECK-SAME: tcrv_rvv.q4_0_schedule.producer = "rvv-q4-0-autotuner"
// CHECK-SAME: tcrv_rvv.q4_0_schedule.vector_register_budget = 32 : i64

// (B) The GEMM op was auto-discovered + stamped through the GEMM descriptor IN
// THE SAME PASS: the single activation_cols M knob + the
// "tcrv_rvv.q4_0_gemm_schedule.*" provenance (vreg_ceiling audit flavor).
// CHECK: tcrv_rvv.q4_0_q8_0_gemm
// CHECK-SAME: activation_cols = 4 : i64
// CHECK-SAME: tcrv_rvv.q4_0_gemm_schedule.producer = "rvv-gemm-m-autotuner"
// CHECK-SAME: tcrv_rvv.q4_0_gemm_schedule.vreg_ceiling = 8 : i64
