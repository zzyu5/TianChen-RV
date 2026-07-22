// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 M2-后 iq4_xs codebook front-door: the monolithic weft_rvv.repack_gemm_iq4_xs_q8_K op is
// RETIRED; the ggml iq4_xs 16x1-REPACKED GEMM (prefill) hot kernel -- the SUPER-BLOCK
// CODEBOOK prefill sibling of iq4_nl -- is now CONSTRUCTED as the typed
// weft_rvv.typed_repack_gemm_loop_body region (fold_model "codebook_superblock_signed6_no_min")
// carrying the SHARED weft_rvv.repack_gemm_codebook_core integer-core brick (decode_model
// "iq4_xs" + the SAME 16-entry NON-LINEAR int8 codebook). The K-quant 6-bit SIGNED
// per-sub-block scale facts (scales_l LOW pair @64, scales_h HIGH 2-bit @32, n_subblocks 8)
// ride on the loop body op's OPTIONAL super-block attrs. The block_index + strip_row_offset
// tied core brick is the anti-bypass surface; verifyRepackCodebookCoreCommon is fail-closed
// (I7) and accepts decode_model "iq4_xs" alongside "iq4_nl".

module {
// CHECK-LABEL: weft.exec.kernel @repack_gemm_codebook_iq4xs_accepts_superblock_abi
  weft.exec.kernel @repack_gemm_codebook_iq4xs_accepts_superblock_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // CHECK: weft_rvv.typed_repack_gemm_loop_body
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock.fp16-signed6-scale-codebook-4col-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_superblock_signed6_no_min", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // CHECK: weft_rvv.repack_gemm_codebook_core
          %sumi:4 = weft_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_xs", weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

// The SHARED codebook GEMM core verifier accepts ONLY decode_model "iq4_nl" / "iq4_xs"
// (fail-closed, I7): an unknown decode_model is rejected.
module {
  weft.exec.kernel @repack_gemm_codebook_rejects_unknown_decode_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock.fp16-signed6-scale-codebook-4col-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_superblock_signed6_no_min", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // expected-error @+1 {{only accepts decode_model "iq4_nl"}}
          %sumi:4 = weft_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_zz", weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}
