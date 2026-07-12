// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M4 iq2_xs GRID PREFILL CONSTRUCTION -- the prefill (M>>1 GEMM) sibling of the iq2_xs
// dual-ls grid decode construct proof. The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (iq2_xs grid / m_regime = "prefill") is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: REPACK is selected (block_dot_compute_heavy
// + VLEN128), and the C1 bridge lowerToRepackGemmGrid CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model "grid_sign_dualscale_eighth")
// carrying the SINGLE weft_rvv.repack_gemm_grid_core brick (decode_model "iq2_xs",
// block_index + strip_row_offset double anti-bypass), reconstructing the block_iq2_xsx16
// x16 weight facts + the INTERLEAVED block_q8_Kx4 activation facts (1168/16), AND
// MATERIALIZING the nr/bs GEMM ABI values the abstract op does not carry. Ships PLAIN
// (untiled): iq2_xs sits at the <=32-vreg cliff. NO perf/e2e claim -- lit-emitted.

module {
  weft.exec.kernel @ggml_repack_gemm_iq2_xs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq2_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq2_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq2_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT dual grid PREFILL request: PLAIN block_iq2_xs (stride 74, qs @2) x
        // PLAIN block_q8_K (stride 292), qk 256, m_regime = "prefill",
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs plane and
        // NO nr/bs -- the compiler RECONSTRUCTS the planes + MATERIALIZES the GEMM ABI values.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq2_xs", scale_model = "superblock-d.fp16-grid-sign-dualscale-nomin-eighth", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 74 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the dual grid typed_repack GEMM region
// (fold_model grid_sign_dualscale_eighth) carrying the grid GEMM core brick (decode_model
// iq2_xs), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: weft_rvv.runtime_abi_value
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "grid_sign_dualscale_eighth"
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// The GEMM grid core brick has BOTH the block_index and strip_row_offset anti-bypass ties.
// CONSTRUCT: weft_rvv.repack_gemm_grid_core
// CONSTRUCT-SAME: decode_model = "iq2_xs"
// CONSTRUCT-SAME: weight_sign_byte_offset = 1312

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq2_xs GEMM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemm-iq2-xs-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq2_xs_q8_K_kernel_ggml_repack_gemm_iq2_xs_q8_K(
// The FIXED 512-entry grid + DERIVED signs64 plane decls (RECONSTRUCTED by the compiler).
// CHECK: verbatim "static const int64_t weft_iq2xs_grid[512] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2xs_signs64[1024] = {1
// The interleaved block_q8_Kx4 activation stride 1168; the weight stride 1824.
// CHECK: literal "1824"
// CHECK: literal "1168"
// The REAL memory grid + sign GATHER (AMORTIZED across the 4 interleaved columns).
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The grid + sign decode are MEMORY gathers, NOT a register vrgather; iq2_xs is scale-ONLY:
// NO min term (no vfnmsac), NO cross-lane reduction wall (no vredsum).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
