// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch2 CONSTRUCTION (GEMM prefill finale) -- the ternary tq1_0 (BASE-3)
// sibling of the tq2_0 rvv-emit-quant-contraction-tq2-0-repack-gemm-prefill-vlen128
// proof. The abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction op (ternary
// tq1_0 / PREFILL) is AUTO-LOWERED at rv64gcv: the in-compiler selection picks REPACK
// (facts + VLEN128) and -- because m_regime == "prefill" AND the committed scale_model
// is the ternary tq1_0 base-3 WHAT -- the C1 bridge lowerToRepackGemmTernary (shared with
// tq2_0, parameterized by the per-family TernaryDecodeFacts) CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model "ternary_single_fp16_scale")
// carrying the SINGLE weft_rvv.repack_gemm_ternary_core integer-core brick (decode_model
// "tq1_0"), reconstructing the block_tq1_0x16 weight facts 864/16/32 + the base-3 qh
// SECOND-plane offset 800 + the INTERLEAVED block_q8_Kx4 activation facts 1168/4/16,
// half_lanes=8 => mf2, columnsPerPass==4, and MATERIALIZES the two GEMM ABI values nr/bs
// the abstract op does not carry. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemm_tq1_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT ternary PREFILL request: PLAIN block_tq1_0 (stride 54, base-3 qs
        // plane at +0) x PLAIN block_q8_K (stride 292), qk 256, scale_model = the base
        // ternary tq1_0 WHAT, m_regime = prefill, block_dot_compute_heavy = true (routes
        // REPACK => GEMM).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "tq1_0", scale_model = "superblock-d.fp16-single-scale-base3-ternary-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the ternary tq1_0 typed_repack GEMM
// region (fold_model ternary_single_fp16_scale) carrying the ternary GEMM core brick
// (decode_model tq1_0), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that
// order: activation_block_stride < activation_interleave < fold_model < half_lanes <
// weft_rvv.* audit attrs < weight_block_stride < weight_qh_byte_offset.
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "ternary_single_fp16_scale"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 864
// CONSTRUCT-SAME: weight_qh_byte_offset = 800
// The in-region block_index + strip_row_offset tied ternary GEMM integer-core BRICK.
// CONSTRUCT: weft_rvv.repack_gemm_ternary_core
// CONSTRUCT-SAME: decode_model = "tq1_0"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the base-3 ternary GEMM kernel (byte-exact to the
// retired direct emitter -- the SAME emit as rvv-to-emitc-repack-gemm-tq1-0-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_ternary_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_tq1_0_q8_K_kernel_ggml_repack_gemm_tq1_0_q8_K(
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base
// vx + x*nb*864 (block_tq1_0x16).
// CHECK: literal "1168"
// CHECK: literal "864"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The tq1_0 BASE-3 TERNARY weight assembly SHARED across columns: vmul_vx_u8 (byte*pow3),
// vwmulu_vx_u16 (*3), vsrl_vx_u16 (>>8), vncvt, reinterpret to i8, vadd_vx_i8 by -1.
// CHECK: call_opaque "__riscv_vmul_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2"
// The per-column integer dot + i16->i32 widening fold, the qh SECOND plane at +800, and
// the SINGLE-scale float fold.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: literal "800"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NO cross-lane reduction wall; the LINEAR ternary fold has NO min term (no vfnmsac)
// and NO per-sub-block scale (no vwmacc_vv_i32).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
