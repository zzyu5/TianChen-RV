// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format4 CONSTRUCTION (PREFILL/GEMM) -- the K-quant q5_K sibling of the q4_K
// rvv-emit-quant-contraction-gemm-prefill proof, the THIRD MIN-fold K-quant (after
// q4_K/q2_K) and the LAST K-quant repack sibling, COMPLETING the K-quant repack family. The
// abstract tcrv_rvv.quant_contraction op (K-quant q5_K / prefill) is AUTO-LOWERED at
// rv64gcv: the selection reads block_dot_compute_heavy = true + the derived VLEN (128) and
// picks REPACK; m_regime = prefill => lowerToRepackGemmKQuant (parameterized by
// kQ5KDecodeFacts, hasMin=true -- REUSING the q4_K dual d/dmin + bsums-min fold WHOLE, ZERO
// framework re-pay) CONSTRUCTS the typed tcrv_rvv.typed_repack_gemm_loop_body REGION
// (fold_model "kquant_dmin_bsums_min", SHARED with q4_K) carrying the SINGLE
// tcrv_rvv.repack_gemm_kquant_core brick (decode_model "q5_K"), reconstructs the
// block_q5_Kx16 weight facts (2816/768, dmin @32, 6-bit scales @64, the qh 5th-bit plane
// @256 [the SHARED qh slot, here on a MIN fold]) + the INTERLEAVED block_q8_Kx4 activation
// facts (1168/16, bsums @1040), n_subblocks 8, and MATERIALIZES the nr/bs GEMM ABI values
// the abstract op does not carry. The constructed GEMM body ships S6-TILED (the register-
// cliff lever transfers WHOLE like q2_K -- q5_K SHARES the q4_K min fold; the qh inject is
// orthogonal to the tiling). NO perf/e2e claim.

module {
  tcrv.exec.kernel @ggml_repack_gemm_q5_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT K-quant q5_K PREFILL request: PLAIN block_q5_K (stride 176, qs @48,
        // qh @16) x PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q5_K
        // WHAT, m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q5_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 176 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 48 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q5_K typed_repack GEMM
// region (fold_model kquant_dmin_bsums_min, SHARED with q4_K) carrying the K-quant GEMM
// core brick (decode_model q5_K), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically. The q4_K MIN structure (dmin + bsums) AND the
// q5_K qh 5th-bit plane are BOTH stamped.
// CONSTRUCT: tcrv_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_bsums_byte_offset = 1040
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: activation_quant_byte_offset = 16
// CONSTRUCT-SAME: fold_model = "kquant_dmin_bsums_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5"
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 2816
// CONSTRUCT-SAME: weight_dmin_byte_offset = 32
// CONSTRUCT-SAME: weight_qh_byte_offset = 256
// CONSTRUCT-SAME: weight_quant_byte_offset = 768
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// The in-region block_index + strip_row_offset tied K-quant GEMM integer-core BRICK.
// CONSTRUCT: tcrv_rvv.repack_gemm_kquant_core
// CONSTRUCT-SAME: decode_model = "q5_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the S6-TILED q5_K GEMM kernel (byte-exact to the plain
// untiled q5_K emit -- the SAME emit as rvv-to-emitc-repack-gemm-q5-K-q8-K).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemm_kquant_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base
// vx + x*nb*2816 (block_q5_Kx16).
// CHECK: literal "1168"
// CHECK: literal "2816"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The S6 stack panels: the decoded 6-bit scale/min i16 strips are staged (vse16/vle16) and
// the idle per-column i32 MIN accumulator is staged (vse32/vle32) OFF the hot live set.
// CHECK: call_opaque "__riscv_vse16_v_i16m1"
// CHECK: call_opaque "__riscv_vle16_v_i16m1"
// The q5_K 5-bit weight assembly (SHARED across the 4 columns): the q4_K nibble PLUS a
// SECOND vle8 loading the qh strip, vand 0x01 masking the selected per-sub-block bit (a
// mask q4_K NEVER emits), vsll 4, vor merging onto the nibble -> a 5-bit value in [0,31].
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// The per-column lane-wise integer dot (vwmacc_vx) + scale-weighted i32 promote (vwmacc_vv).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block dual fold per column: vfmul_vf (d_x*d_y_c), vfcvt, vfmacc (main), then
// vfnmsac (MIN term reloaded from the bsums stack panel).
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
