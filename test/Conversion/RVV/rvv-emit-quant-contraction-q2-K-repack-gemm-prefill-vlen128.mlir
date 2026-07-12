// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format2 CONSTRUCTION (PREFILL/GEMM) -- the K-quant q2_K sibling of the
// q4_K rvv-emit-quant-contraction-q4-K-repack-gemm-prefill proof, REUSING the front-door
// q4_K MIN fold with ZERO framework generalization. The abstract weft_rvv.quant_contraction
// op (K-quant q2_K / prefill) is AUTO-LOWERED at rv64gcv: the selection reads
// block_dot_compute_heavy = true + the derived VLEN (128) and picks REPACK; m_regime =
// prefill => lowerToRepackGemmKQuant (parameterized by kQ2KDecodeFacts, hasMin=true)
// CONSTRUCTS the typed weft_rvv.typed_repack_gemm_loop_body REGION (fold_model
// "kquant_dmin_bsums_min", SHARED with q4_K) carrying the SINGLE
// weft_rvv.repack_gemm_kquant_core brick (decode_model "q2_K"), reconstructs the
// block_q2_Kx16 weight facts (1344/320, dmin @32, 4-bit packed scales @64) + the
// INTERLEAVED block_q8_Kx4 activation facts (1168/16, bsums @1040), n_subblocks 16, and
// MATERIALIZES the nr/bs GEMM ABI values the abstract op does not carry. The constructed
// GEMM body is S6-TILED (h-strip output tile + decoded 4-bit scale/min stack-panel + idle
// i32 MIN accumulator stack-panel + on-demand d/dmin; SSA-register accumulators, the
// q4_K register-cliff lever transferred WHOLE). NO perf/e2e claim.

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
        // The ABSTRACT K-quant q2_K PREFILL request: PLAIN block_q2_K (stride 84, qs @16)
        // x PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q2_K WHAT,
        // m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q2_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q2_K typed_repack GEMM
// region (fold_model kquant_dmin_bsums_min, SHARED with q4_K) carrying the K-quant GEMM
// core brick (decode_model q2_K), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically. q2_K has the dmin+bsums MIN structure.
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_bsums_byte_offset = 1040
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: activation_quant_byte_offset = 16
// CONSTRUCT-SAME: fold_model = "kquant_dmin_bsums_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 16
// CONSTRUCT-SAME: scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1344
// CONSTRUCT-SAME: weight_dmin_byte_offset = 32
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// CONSTRUCT-NOT: weight_qh_byte_offset
// The in-region block_index + strip_row_offset tied K-quant GEMM integer-core BRICK.
// CONSTRUCT: weft_rvv.repack_gemm_kquant_core
// CONSTRUCT-SAME: decode_model = "q2_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the S6-tiled q2_K GEMM kernel (byte-exact to the
// retired direct emitter -- the SAME emit as rvv-to-emitc-repack-gemm-q2-K-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base
// vx + x*nb*1344 (block_q2_Kx16).
// CHECK: literal "1168"
// CHECK: literal "1344"
// The per-column f32m2 accumulators (columnsPerPass == 4, S1 h-strip tiled).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The q2_K 4-bit packed scale/min unpack staged to the int16_t stack panel (vand 0x0F /
// vsrl 4 / vzext / vse16), reused across the 4 columns.
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vse16_v_i16m1"
// The idle i32 MIN accumulator staged to the int32_t bsums stack panel (vse32).
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// The SHARED 2-bit weight assembly reused across the 4 columns: vand 0x03, reinterpret.
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The per-column lane-wise integer dot (vwmacc_vx) + scale-weighted i32 promote (vwmacc_vv).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block per-column dual d/dmin fold (on-demand widen): vfmacc main + vfnmsac MIN.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
