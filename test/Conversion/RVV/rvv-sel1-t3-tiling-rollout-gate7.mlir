// RUN: weft-opt %s -split-input-file --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s
// RUN: weft-opt %s -split-input-file --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=NOSTATIC

// [G3 主线C / SEL-1] T3 gate-(7) ALL-FORMAT rollout: the SP4 (tiled-vs-plain) output-
// tiling choice is a RUNTIME CAPABILITY-KEYED selection at EVERY repack GEMM leaf, keyed
// on the BOTTLENECK SHAPE derived from fold_model (a structure / capability fact), NEVER
// the format name. This file proves BOTH attribution paths the [SEL-1] design demands:
//
//   MEASURED (the 5 K-quant are offline-seeded on the rvv/VLEN128 board, declared-
//   instance-hash 3cd23a4e...): the memoized-argmin winner is stamped reason=measured.
//     * q6_K / q3_K (fold_model "kquant_single_scale_no_min", weight-reconstruction-
//       bound): the S6 register-cliff lever is a NULL -- measurement itself says "do NOT
//       tile", so the HONEST measured winner is PLAIN. This weight-bound measured
//       fallback IS part of the paper claim (a measured NULL is a first-class datapoint,
//       not a missing win). q6_K shown here; q4_K measured shown in the sibling gate-7.
//     * q2_K / q5_K (fold_model "kquant_dmin_bsums_min", min-fold register cliff): S6
//       reaches the <=32-vreg cliff, so the measured winner is S6Tiled.
//
//   PRIOR (cold start, NO offline seed => the [XFER-1] capability prior keyed on the
//   shape): q4_0 (flat "lane_wise_vector_scale") + iq4_nl (codebook
//   "codebook_flat_single_scale") both classify AlreadyLean => PLAIN, reason=prior.
//
// NO wired path ever stamps static_order (the capability-blind fallback): every leaf
// here sits on a capability-afforded board (VLEN128, 32 vregs), so the feasible set is
// {plain, s6_tiled} and the decision is always capability/measurement-DERIVED.
// NOSTATIC-NOT: static_order

// ===================== q6_K: MEASURED -> PLAIN (weight-bound honest fallback) ========
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q6_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_single_scale_no_min"
// [M1c] loop-order axis: keyed on the layout stride fact (weight x16 panel > activation
// panel => col_outer); NO loop-order A/B seed here (only q4_K is) => cold-start prior.
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "measured"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q6_K{{.*}}reason{{.*}}measured
// CHECK-SAME: weft_rvv.tiling_variant = "plain"
module {
  weft.exec.kernel @ggml_repack_gemm_q6_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q6_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q6_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q6_K", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// ===================== q2_K: MEASURED -> S6Tiled (min-fold register cliff) ===========
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q2_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_dmin_bsums_min"
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "measured"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q2_K{{.*}}reason{{.*}}measured
// CHECK-SAME: weft_rvv.tiling_variant = "s6_tiled"
module {
  weft.exec.kernel @ggml_repack_gemm_q2_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q2_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q2_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q2_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// ===================== q5_K: MEASURED -> S6Tiled (min-fold survives qh plane) ========
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q5_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_dmin_bsums_min"
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "measured"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q5_K{{.*}}reason{{.*}}measured
// CHECK-SAME: weft_rvv.tiling_variant = "s6_tiled"
module {
  weft.exec.kernel @ggml_repack_gemm_q5_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q5_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 176 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 48 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// ===================== q4_0: PRIOR -> PLAIN (flat already-lean cold start) ===========
// CHECK-LABEL: weft.exec.variant @ggml_gemm_q4_0_q8_0
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "lane_wise_vector_scale"
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q4_0{{.*}}reason{{.*}}prior
// CHECK-SAME: weft_rvv.tiling_variant = "plain"
module {
  weft.exec.kernel @ggml_gemm_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_gemm_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_gemm_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_gemm_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// ===================== iq4_nl: PRIOR -> PLAIN (codebook already-lean cold start) =====
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_iq4_nl_q8_0
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "codebook_flat_single_scale"
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "prior"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}iq4_nl{{.*}}reason{{.*}}prior
// CHECK-SAME: weft_rvv.tiling_variant = "plain"
module {
  weft.exec.kernel @ggml_repack_gemm_iq4_nl_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq4_nl", scale_model = "flat.fp16-single-scale-codebook-nomin", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
