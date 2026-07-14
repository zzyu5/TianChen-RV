// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format3 CONSTRUCTION (PREFILL/GEMM) -- the K-quant q3_K sibling of the q6_K
// rvv-emit-quant-contraction-q6-K-repack-gemm-prefill proof, the SECOND NO-MIN K-quant
// fold (after q6_K) and the LAST K-quant repack sibling. The abstract
// weft_rvv.quant_contraction op (K-quant q3_K / prefill) is AUTO-LOWERED at rv64gcv: the
// selection reads block_dot_compute_heavy = true + the derived VLEN (128) and picks
// REPACK; m_regime = prefill => lowerToRepackGemmKQuant (parameterized by kQ3KDecodeFacts,
// hasMin=false -- REUSING the q6_K no-min fold, ZERO framework re-pay) CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model "kquant_single_scale_no_min")
// carrying the SINGLE weft_rvv.repack_gemm_kquant_core brick (decode_model "q3_K"),
// reconstructs the block_q3_Kx16 weight facts (1824/800, hmask @288, SIGNED scales @32) +
// the INTERLEAVED block_q8_Kx4 activation facts (1168/16, NO bsums), n_subblocks 16, and
// MATERIALIZES the nr/bs GEMM ABI values the abstract op does not carry. The constructed
// GEMM body ships PLAIN/UNTILED (S6 output tiling is a family-lever NULL for
// weight-reconstruction-bound q3_K -- like q6_K, the q4_K register-cliff lever does NOT
// transfer -- so the GEMM ships plain per 更简单者胜). NO perf/e2e claim.

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
        // The ABSTRACT K-quant q3_K PREFILL request: PLAIN block_q3_K (stride 110, hmask @0)
        // x PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q3_K WHAT,
        // m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q3_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q3_K typed_repack GEMM
// region (fold_model kquant_single_scale_no_min) carrying the K-quant GEMM core brick
// (decode_model q3_K), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically. NO dmin / NO bsums (single-accumulator no-min).
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: activation_quant_byte_offset = 16
// CONSTRUCT-SAME: fold_model = "kquant_single_scale_no_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 16
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// CONSTRUCT-SAME: weight_qh_byte_offset = 288
// CONSTRUCT-SAME: weight_scales_byte_offset = 32
// CONSTRUCT-NOT: weight_dmin_byte_offset
// The in-region block_index + strip_row_offset tied K-quant GEMM integer-core BRICK.
// CONSTRUCT: weft_rvv.repack_gemm_kquant_core
// CONSTRUCT-SAME: decode_model = "q3_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the PLAIN q3_K GEMM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemm-q3-K-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base
// vx + x*nb*1824 (block_q3_Kx16).
// CHECK: literal "1168"
// CHECK: literal "1824"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The q3_K SIGNED int8 per-sub-block scale reused across the 4 columns: vle8_v_i8 + vsext.
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The SHARED 3-bit subtractive qs|hmask weight assembly reused across the 4 columns
// (NATIVE-MASK KNEST via the SHARED [QH-MASK] helper): vle8, 2-bit low plane vand 0x03
// reinterpreted to signed i8, then the SINGLE hmask high bit tested IN PLACE by
// vand(1<<p) + vmseq==0 with the -4 SUBTRACTIVE bias FUSED into ONE vadd_vx_i8mf2_mu --
// byte-exact to the retired vsll 2 | vor | vsub 4 chain (the q3_K GEVM sibling proves it).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vmseq_vx_u8mf2_b16"
// CHECK: literal "-4"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2_mu"
// The per-column lane-wise integer dot (vwmacc_vx) + scale-weighted i32 promote (vwmacc_vv).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block SINGLE fold per column: vfmacc the ONE main term (NO vfnmsac min term).
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
// q3_K has NO min term: NO vfnmsac.
// CHECK-NOT: vfnmsac

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
