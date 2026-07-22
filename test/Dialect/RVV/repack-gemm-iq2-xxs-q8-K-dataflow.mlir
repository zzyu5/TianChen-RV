// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 M4 iq2-grid front-door (first cell iq2_xxs): the monolithic
// weft_rvv.repack_gemm_iq2_xxs_q8_K op is RETIRED; the ggml iq2_xxs 16x1-REPACKED PREFILL
// GEMM hot kernel is now CONSTRUCTED as the typed weft_rvv.typed_repack_gemm_loop_body
// region (fold_model "grid_sign_single_scale_eighth") carrying the
// weft_rvv.repack_gemm_grid_core integer-core brick (decode_model "iq2_xxs" + the grid /
// ls / sign byte offsets + n_subblocks), block_index + strip_row_offset tied (double
// anti-bypass). The core verifier (verifyRepackGridCoreCommon) is fail-closed (I7) on a
// wrong decode_model / a non-positive byte offset.

module {
// CHECK-LABEL: weft.exec.kernel @repack_gemm_grid_iq2xxs_accepts_16x1_abi
  weft.exec.kernel @repack_gemm_grid_iq2xxs_accepts_16x1_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_grid_iq2xxs_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: weft_rvv.typed_repack_gemm_loop_body
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth-4col", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // CHECK: weft_rvv.repack_gemm_grid_core
          %sumi:4 = weft_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @repack_gemm_grid_rejects_unknown_decode_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_grid_rejects_unknown_decode_model", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth-4col", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // C4a: accepted set = the CLOSED weft::GridDecodePlan registry (see the GEVM
          // sibling). Fail-closed UNCHANGED: unregistered decode_model still REJECTED.
          // expected-error @+1 {{only accepts a decode_model registered in the closed grid decode plan registry}}
          %sumi:4 = weft_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_zz", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @repack_gemm_grid_rejects_nonpositive_ls_offset {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_grid_rejects_nonpositive_ls_offset", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth-4col", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // expected-error @+1 {{requires positive grid-index / ls-scale / sign-selector / activation-quant byte offsets and n_subblocks}}
          %sumi:4 = weft_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 0 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}
