// RUN: weft-opt %s -split-input-file --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s
// RUN: weft-opt %s -split-input-file --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=NOSTATIC

// [SEL-3 / A4a] Measurement-boundary falsifier. Historical SP4 A/B rows still exist as
// evidence, but SP4 no longer performs measured ranking because each shape currently has
// one real body. These five formerly seeded leaves must therefore stamp only_feasible and
// the shape-realizable singleton. The independent loop-order axis remains a genuine
// two-body choice: q4_K still receives col_outer/reason=measured from the qualified-view
// seed. This catches both regressions: reviving an SP4 fake candidate, or accidentally
// deleting the real loop-order measurement path. No wired leaf stamps static_order.
// NOSTATIC-NOT: static_order

// ===================== q4_K: singleton s6_tiled + measured col_outer =================
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q4_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_dmin_bsums_min"
// CHECK-SAME: weft_rvv.loop_order = "col_outer"
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "measured"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q4_K{{.*}}reason{{.*}}only_feasible
// CHECK-SAME: weft_rvv.tiling_variant = "s6_tiled"
module {
  weft.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -----

// ===================== q2_K: singleton s6_tiled (min-fold register cliff) =============
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q2_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_dmin_bsums_min"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q2_K{{.*}}reason{{.*}}only_feasible
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

// ===================== q5_K: singleton s6_tiled (min-fold survives qh plane) ===========
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q5_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_dmin_bsums_min"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q5_K{{.*}}reason{{.*}}only_feasible
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

// ===================== q6_K: singleton plain (weight-bound body) ======================
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q6_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_single_scale_no_min"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q6_K{{.*}}reason{{.*}}only_feasible
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

// ===================== q3_K: singleton plain (weight-bound body) ======================
// The historical +8% tiled observation remains a null-lever evidence row; it does not
// make an unrealized tiled body a compiler candidate.
// CHECK-LABEL: weft.exec.variant @ggml_repack_gemm_q3_K_q8_K
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// CHECK-SAME: fold_model = "kquant_single_scale_no_min"
// CHECK-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// CHECK-SAME: weft_rvv.tiling_selection_record = "{{.*}}kernel{{.*}}q3_K{{.*}}reason{{.*}}only_feasible
// CHECK-SAME: weft_rvv.tiling_variant = "plain"
module {
  weft.exec.kernel @ggml_repack_gemm_q3_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q3_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
