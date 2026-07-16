// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-4 iq3_xxs PREFILL GEMM CONSTRUCTION -- the prefill half of the iq3_xxs landing. The
// abstract weft_rvv.quant_contraction (iq3_xxs grid / m_regime = "prefill") is AUTO-LOWERED
// at rv64gcv: selection picks REPACK, and lowerToRepackGemmGrid CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model
// "grid_sign_dual_entry_single_scale_quarter") carrying the weft_rvv.repack_gemm_grid_core
// brick (decode_model "iq3_xxs"), reconstructing the block_iq3_xxsx16 x16 weight facts
// 1696/160/32/1184 AND the INTERLEAVED block_q8_Kx4 activation facts (stride 1168, quants
// @16 as pos*4+c -- NO bsums, its single-accumulator fold has no delta term), plus
// MATERIALIZING the two GEMM ABI values (nr, bs) the abstract op does not carry.
//
// This is the "front door can construct a repack GEMM op" half of the C4a-4 acceptance:
// registry row + front door + BOTH emitter leaves + byte-exact oracle, together. Without
// the emitter leaf this construction would be exactly the "verifier accepts but nobody can
// lower" state C4a refused to ship.
//
// NO perf/e2e claim: C4a-4 never touched a board. Numeric correctness is gated byte-exactly
// (GEMM interleaved addressing included) by tools/oracle-repack/oracle_repack_iq3_xxs.cpp.

module {
  weft.exec.kernel @ggml_repack_gemm_iq3_xxs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq3_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq3_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq3_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT dual-entry grid PREFILL request: PLAIN block_iq3_xxs (stride 98 =
        // fp16 d + 3*(QK_K/8) = 96 qs bytes, qs @2) x
        // PLAIN block_q8_K (stride 292), qk 256, m_regime = "prefill",
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs plane and
        // NO nr/bs -- the compiler RECONSTRUCTS the planes + MATERIALIZES the GEMM ABI values.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq3_xxs", scale_model = "superblock-d.fp16-grid-sign-dual-entry-4bit-scale-nomin-quarter", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 98 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the dual-entry grid typed_repack GEMM
// region carrying the grid core brick, NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// (CONSTRUCT-SAME matches left-to-right on one line, so these follow the op's own
// alphabetical attribute order.) The INTERLEAVED block_q8_Kx4 GEMM activation stride comes
// first, then the 1696 repacked weight stride -- iq2_xxs's 1184 plus the extra 512 B its
// DOUBLE-width grid-index strip needs.
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "grid_sign_dual_entry_single_scale_quarter"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: weight_block_stride = 1696
// NO bsums attr either side: the SignScaleStore fold has no delta term to feed.
// CONSTRUCT-NOT: activation_bsums_byte_offset
// CONSTRUCT: weft_rvv.repack_gemm_grid_core
// CONSTRUCT-SAME: decode_model = "iq3_xxs"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 1184

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq3_xxs prefill GEMM body via the NEW dual-entry
// leaf -- the half that makes the registry row honest.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq3_xxs_q8_K_kernel_ggml_repack_gemm_iq3_xxs_q8_K(
// The FIXED 256-entry uint32 grid + the DERIVED signs64 plane, both RECONSTRUCTED.
// CHECK: verbatim "static const uint32_t weft_iq3xxs_grid[256] = {0x04040404U
// CHECK: verbatim "static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1,
// The table POINTERS come from the registry row as DATA, and must name the tables declared
// above; the trailing quote makes each match exact.
// CHECK: literal "weft_iq3xxs_grid"
// CHECK: literal "weft_iq2xxs_signs64"
// The repacked weight stride 1696 + the INTERLEAVED activation stride 1168.
// CHECK: literal "1696"
// CHECK: literal "1168"
// The REAL memory grid GATHER (u8 index -> vzext -> vsll -> vluxei16).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// Sign fold onto the grid byte + the i32 dot + the SINGLE per-sub-block ls vmacc.
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// THE DUAL-ENTRY SIGNATURE: byte offset 1176, the LAST grid-index strip offset the
// dual-entry nest reaches -- gridIdxOffset + (ib*8 + grp*2 + e)*16 + h*8 at ib=7, grp=3,
// e=1, h=1 = 160 + 1008 + 8. The single-base nest strides by (ib*4 + grp)*16 and tops out
// at 664, so it cannot emit this. See the GEVM sibling for the injection that proves it
// goes red, and for the shift-literal CHECK that was tried first and discarded as vacuous
// (`literal "2"` occurs throughout the mis-lowered output too).
// CHECK: literal "1176"
// The SignScaleStore fold + ggml's 0.25f store constant (NOT the iq2 rows' 0.125f).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.25f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The store-side step comment must name the constant it is attached to: this row stores
// 0.25f, so the marker is quarter_scale, never the iq2 rows' eighth_scale.
// CHECK-NOT: callee=eighth_scale

// The grid decode is a MEMORY gather, NOT a register vrgather; the block-as-lane repack
// erases the cross-lane reduction wall; the dual-entry fold has NO min.
// NOWALL-NOT: vrgather
