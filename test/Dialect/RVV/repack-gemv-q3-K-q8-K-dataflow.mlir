// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G3 主线A T3 format3: the monolithic weft_rvv.repack_gemv_q3_K_q8_K op is RETIRED; the
// ggml q3_K 16x1-REPACKED GEVM (decode) hot kernel is now CONSTRUCTED as the typed
// weft_rvv.typed_repack_gemv_loop_body region (fold_model "kquant_single_scale_no_min",
// SHARED with q6_K) carrying the weft_rvv.repack_gemv_kquant_core integer-core brick
// (decode_model "q3_K"). q3_K is q6_K's no-min structural cousin: a 3-bit SIGNED weight
// assembled from a 2-bit qs low plane + a SINGLE 1-bit hmask high plane, SUBTRACTIVE (bit
// CLEAR -> subtract 4; `((qs&3)|(hbit<<2)) - 4`), 16 SIGNED 6-bit per-sub-block scales
// (pre-unpacked + -32-biased at repack), and NO per-sub-block min term (a SINGLE
// accumulator, the -4 bias inside each weight lane). The region consumes the repacked
// block_q3_Kx16 weights (vx, byte stride (n/QK_K)*1824) + ONE plain block_q8_K activation
// stream (vy, stride 292) + the fp32 output (s) + n + nc, carries the q3_K super-block
// facts (qs at +800, hmask at +288 [the SHARED weight_qh_byte_offset slot], signed scales
// at +32, 16 sub-blocks) as OPTIONAL loop-body attrs (I4 mirror; NO dmin/bsums -- no-min),
// and accumulates LANE-WISE via vwmacc (NO cross-lane vredsum). The block_index-tied core
// brick (decode_model "q3_K") is the anti-bypass surface.

module {
// CHECK-LABEL: weft.exec.kernel @repack_gemv_q3k_region_accepts_16x1_abi
  weft.exec.kernel @repack_gemv_q3k_region_accepts_16x1_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // CHECK: weft_rvv.typed_repack_gemv_loop_body
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "kquant_single_scale_no_min", main_term_form = "unrolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // CHECK: weft_rvv.repack_gemv_kquant_core
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

// The K-quant core brick REJECTS an unrecognized decode_model (fail-closed I7): only
// q4_K / q2_K (min fold) / q6_K / q3_K (no-min fold) are accepted.
module {
  weft.exec.kernel @repack_gemv_q3k_rejects_bad_decode_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "kquant_single_scale_no_min", main_term_form = "unrolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // expected-error @+1 {{only accepts decode_model}}
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q7_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}
