// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch1 CONSTRUCTION (GEMM prefill finale) -- the ternary tq2_0 sibling of
// the q4_0 rvv-emit-quant-contraction-q4-0-repack-gemm-prefill-vlen128 proof. The
// abstract, algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op (ternary tq2_0 /
// PREFILL) is AUTO-LOWERED at rv64gcv: the in-compiler selection picks REPACK (facts
// + VLEN128) and -- because m_regime == "prefill" AND the committed scale_model is
// the ternary WHAT -- the C1 bridge lowerToRepackGemmTernary CONSTRUCTS the typed
// tcrv_rvv.typed_repack_gemm_loop_body REGION (fold_model "ternary_single_fp16_scale")
// carrying the SINGLE tcrv_rvv.repack_gemm_ternary_core integer-core brick
// (decode_model "tq2_0"), reconstructing the block_tq2_0x16 weight facts 1056/16/32 +
// the INTERLEAVED block_q8_Kx4 activation facts 1168/4/16, half_lanes=8 => mf2,
// columnsPerPass==4, and MATERIALIZES the two GEMM ABI values nr/bs the abstract op
// does not carry. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_repack_gemm_tq2_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT ternary PREFILL request: PLAIN block_tq2_0 (stride 66) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the base ternary tq2_0 WHAT,
        // m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "tq2_0", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the ternary typed_repack GEMM
// region (fold_model ternary_single_fp16_scale) carrying the ternary GEMM core brick
// (decode_model tq2_0), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track
// that order: activation_block_stride < activation_interleave < fold_model <
// half_lanes < tcrv_rvv.* audit attrs < weight_block_stride.
// CONSTRUCT: tcrv_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "ternary_single_fp16_scale"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1056
// The in-region block_index + strip_row_offset tied ternary GEMM integer-core BRICK.
// CONSTRUCT: tcrv_rvv.repack_gemm_ternary_core
// CONSTRUCT-SAME: decode_model = "tq2_0"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the ternary GEMM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemm-tq2-0-q8-K).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemm_ternary_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_tq2_0_q8_K_kernel_ggml_repack_gemm_tq2_0_q8_K(
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base
// vx + x*nb*1056 (block_tq2_0x16).
// CHECK: literal "1168"
// CHECK: literal "1056"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The tq2_0 2-bit TERNARY weight assembly SHARED across columns: vand 0x03, i8, vsub 1.
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The per-column integer dot + i16->i32 widening fold + SINGLE-scale float fold.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
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
