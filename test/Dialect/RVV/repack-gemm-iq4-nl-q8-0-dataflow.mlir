// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 M2 iq4_nl codebook front-door: the monolithic tcrv_rvv.repack_gemm_iq4_nl_q8_0 op is
// RETIRED; the ggml iq4_nl 16x1-REPACKED GEMM (prefill) hot kernel is now CONSTRUCTED as the
// typed tcrv_rvv.typed_repack_gemm_loop_body region (fold_model "codebook_flat_single_scale")
// carrying the tcrv_rvv.repack_gemm_codebook_core integer-core brick (decode_model "iq4_nl" +
// the 16-entry NON-LINEAR int8 codebook). The block_index + strip_row_offset tied core brick
// is the anti-bypass surface; verifyRepackCodebookCoreCommon is fail-closed (I7).

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_codebook_iq4nl_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemm_codebook_iq4nl_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_codebook_iq4nl_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.typed_repack_gemm_loop_body
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "flat.fp16-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_single_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // CHECK: tcrv_rvv.repack_gemm_codebook_core
          %sumi:4 = tcrv_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_nl", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_codebook_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_codebook_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "flat.fp16-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_single_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{currently supports only kind "repack_gemm_codebook_core"}}
          %sumi:4 = tcrv_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "gemm", decode_model = "iq4_nl", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @repack_gemm_codebook_rejects_bad_codebook_size {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_codebook_rejects_bad_codebook_size", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "flat.fp16-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_single_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // expected-error @+1 {{requires a 16-entry non-linear int8 codebook}}
          %sumi:4 = tcrv_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_nl", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, codebook = array<i8: -127, -104, -83, -65>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}
