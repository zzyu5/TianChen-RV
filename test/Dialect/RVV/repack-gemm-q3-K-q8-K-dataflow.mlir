// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 主线A T3 format3: the monolithic tcrv_rvv.repack_gemm_q3_K_q8_K op is RETIRED; the
// ggml q3_K 16x1-REPACKED PREFILL GEMM (M>>1) hot kernel is now CONSTRUCTED as the typed
// tcrv_rvv.typed_repack_gemm_loop_body region (fold_model "kquant_single_scale_no_min",
// SHARED with q6_K) carrying the tcrv_rvv.repack_gemm_kquant_core integer-core brick
// (decode_model "q3_K"). q3_K is q6_K's no-min structural cousin (a 3-bit SUBTRACTIVE
// qs|hmask weight + 16 SIGNED 6-bit scales + SINGLE accumulator, NO min). The region
// consumes the repacked block_q3_Kx16 weights (vx, byte stride (n/QK_K)*1824) + the
// INTERLEAVED block_q8_Kx4 activation stream (vy, stride 1168) + the fp32 output (s) + n +
// nr + nc + bs, carries the q3_K super-block facts (qs at +800, hmask at +288 [the SHARED
// weight_qh_byte_offset slot], signed scales at +32, 16 sub-blocks) as OPTIONAL loop-body
// attrs (I4 mirror; NO dmin/bsums -- no-min), and internalizes the M-row / N-column /
// contraction-block nest. The block_index + strip_row_offset tied core brick (decode_model
// "q3_K") is the anti-bypass surface.

module {
// CHECK-LABEL: tcrv.exec.kernel @repack_gemm_q3k_region_accepts_16x1_abi
  tcrv.exec.kernel @repack_gemm_q3k_region_accepts_16x1_abi {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemm_q3k_region_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        // CHECK: tcrv_rvv.typed_repack_gemm_loop_body
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "kquant_single_scale_no_min"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // CHECK: tcrv_rvv.repack_gemm_kquant_core
          %sumi:4 = tcrv_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}
