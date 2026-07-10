// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 M4 iq2-grid front-door (first cell iq2_xxs): the monolithic
// tcrv_rvv.repack_gemv_iq2_xxs_q8_K op is RETIRED; the ggml iq2_xxs 16x1-REPACKED GEVM
// (decode) hot kernel -- the FIRST SUPER-BLOCK GRID CODEBOOK + SIGN-PLANE decode family --
// is now CONSTRUCTED as the typed tcrv_rvv.typed_repack_gemv_loop_body region (fold_model
// "grid_sign_single_scale_eighth") carrying the tcrv_rvv.repack_gemv_grid_core integer-core
// brick (decode_model "iq2_xxs" + the grid / ls / sign byte offsets + n_subblocks). The
// FIXED 256-entry grid + DERIVED signs64 plane are static const tables, NEVER op attrs. The
// core verifier (verifyRepackGridCoreCommon) is fail-closed (I7) on a wrong kind /
// decode_model / a forbidden attr / a non-positive byte offset; the block_index-tied core
// brick is the anti-bypass surface.

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemv_grid_iq2xxs_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemv_grid_iq2xxs_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_iq2xxs_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.typed_repack_gemv_loop_body
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // CHECK: tcrv_rvv.repack_gemv_grid_core
          %sumi:2 = tcrv_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_grid_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{currently supports only kind "repack_gemv_grid_core"}}
          %sumi:2 = tcrv_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "gemv", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_grid_rejects_unknown_decode_model {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_unknown_decode_model", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{only accepts decode_model "iq2_xxs"}}
          %sumi:2 = tcrv_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_zz", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_grid_rejects_forbidden_min_attr {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_forbidden_min_attr", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{only accepts the bounded grid repack core attributes}}
          %sumi:2 = tcrv_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, weight_dmin_byte_offset = 16 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemv_grid_rejects_nonpositive_sign_offset {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_nonpositive_sign_offset", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{requires positive grid-index / ls-scale / sign-selector / activation-quant byte offsets and n_subblocks}}
          %sumi:2 = tcrv_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}
