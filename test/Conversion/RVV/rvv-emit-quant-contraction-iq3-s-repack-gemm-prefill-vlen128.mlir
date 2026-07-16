// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-5 iq3_s PREFILL GEMM CONSTRUCTION -- the prefill half of the iq3_s landing, the
// SEVENTH and last grid row. The abstract weft_rvv.quant_contraction (iq3_s grid /
// m_regime = "prefill") is AUTO-LOWERED at rv64gcv: selection picks REPACK, and
// lowerToRepackGemmGrid CONSTRUCTS the typed weft_rvv.typed_repack_gemm_loop_body REGION
// (fold_model "grid_sign_dual_entry_single_scale_unit") carrying the
// weft_rvv.repack_gemm_grid_core brick (decode_model "iq3_s"), reconstructing the
// block_iq3_sx16 x16 weight facts 2720/160/32/2208 AND the INTERLEAVED block_q8_Kx4
// activation facts (stride 1168, quants @16 as pos*4+c -- NO bsums, its
// single-accumulator fold has no delta term), plus MATERIALIZING the two GEMM ABI values
// (nr, bs) the abstract op does not carry.
//
// This is the "front door can construct a repack GEMM op" half of the C4a-5 acceptance:
// registry row + front door + BOTH emitter leaves + byte-exact oracle, together. Without
// the emitter leaf this construction would be exactly the "verifier accepts but nobody can
// lower" state C4a refused to ship.
//
// Like its GEVM sibling, this row lowers through the DUAL-ENTRY leaf iq3_xxs paid for,
// selected off the same entryWidth == I32x4 key and parameterized only by the u16 index
// strip (its 9-bit index) and the unit store constant. The amortization across the 4
// activation columns does not touch either: the split is in the WEIGHT decode, which is
// exactly the part the 4 columns share.
//
// NO perf/e2e claim: C4a-5 never touched a board. Numeric correctness is gated byte-exactly
// (GEMM interleaved addressing included) by tools/oracle-repack/oracle_repack_iq3_s.cpp.

module {
  weft.exec.kernel @ggml_repack_gemm_iq3_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq3_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq3_s_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq3_s_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT dual-entry explicit-sign grid PREFILL request: PLAIN block_iq3_s
        // (stride 110 = fp16 d + qs[64] + qh[8] + signs[32] + scales[4], qs @2) x
        // PLAIN block_q8_K (stride 292), qk 256, m_regime = "prefill",
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs plane and
        // NO nr/bs -- the compiler RECONSTRUCTS the planes + MATERIALIZES the GEMM ABI values.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq3_s", scale_model = "superblock-d.fp16-grid-explicitsign-dual-entry-4bit-scale-nomin-unit", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
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
// first, then the 2720 repacked weight stride -- iq3_xxs's 1696 plus the extra 1024 B its
// index strip needs to hold a 9-bit index in a u16 lane instead of a byte.
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "grid_sign_dual_entry_single_scale_unit"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: weight_block_stride = 2720
// NO bsums attr either side: the SignScaleStore fold has no delta term to feed.
// CONSTRUCT-NOT: activation_bsums_byte_offset
// CONSTRUCT: weft_rvv.repack_gemm_grid_core
// CONSTRUCT-SAME: decode_model = "iq3_s"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 2208

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq3_s prefill GEMM body via the NEW dual-entry
// leaf -- the half that makes the registry row honest.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq3_s_q8_K_kernel_ggml_repack_gemm_iq3_s_q8_K(
// The FIXED 512-entry uint32 grid + the DERIVED signs256 plane, both RECONSTRUCTED. 512
// entries (not iq3_xxs's 256) and signs256 (not signs64): iq3_s carries EXPLICIT sign bytes
// with no ksigns selector, so the plane must be indexed by all 256 byte values.
// CHECK: verbatim "static const uint32_t weft_iq3s_grid[512] = {0x01010101U
// CHECK: verbatim "static const int8_t weft_iq2s_signs256[2048] = {1, 1, 1, 1, 1, 1, 1, 1,
// The table POINTERS come from the registry row as DATA, and must name the tables declared
// above; the trailing quote makes each match exact.
// CHECK: literal "weft_iq3s_grid"
// CHECK: literal "weft_iq2s_signs256"
// The repacked weight stride 2720 + the INTERLEAVED activation stride 1168.
// CHECK: literal "2720"
// CHECK: literal "1168"
// The REAL memory grid GATHER. The GRID index strip is u16 (vle16, NO vzext -- 9 bits do
// not fit a byte); the vzext below is the SIGN strip's, which is a byte on every row.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// Sign fold onto the grid byte + the i32 dot + the SINGLE per-sub-block ls vmacc.
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// THE DUAL-ENTRY SIGNATURE: byte offset 2192, the LAST grid-index strip offset the
// dual-entry nest reaches -- gridIdxOffset + ((ib*8 + grp*2 + e)*16 + h*8)*2 at ib=7,
// grp=3, e=1, h=1 = 160 + (1008 + 8)*2 = 2192. (The *2 is this row's u16 index lane; the
// iq3_xxs sibling pins the same discriminator at 1176 with a byte lane.) The single-base
// nest strides by ((ib*4 + grp)*16 + h*8)*2 and tops out at 1168, so it cannot emit this;
// the sign strip starts at 2208, so it cannot come from there either. See the GEVM sibling
// for the injection that proves it goes red (the mis-lowering does NOT fail -- it emits
// confident garbage), and for the shift-literal CHECK that was tried first and discarded as
// vacuous (`literal "2"` occurs throughout the mis-lowered output too).
// CHECK: literal "2192"
// The SignScaleStore fold + iq3_s's store constant. ggml ends `*s = sumf` -- no constant
// at all -- carried as the literal 1.0f the emitter emits; see
// GridDecodePlan::storeScaleLiteral for why it is written that way rather than as an empty
// field. A THIRD distinct store constant in this family, and it cost the fold enum nothing.
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "1.0f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The store-side step comment must name the constant it is attached to: this row stores
// 1, so the marker is unit_scale, never quarter_scale and never eighth_scale.
// CHECK-NOT: callee=eighth_scale

// The grid decode is a MEMORY gather, NOT a register vrgather; the block-as-lane repack
// erases the cross-lane reduction wall; the dual-entry fold has NO min.
// NOWALL-NOT: vrgather
