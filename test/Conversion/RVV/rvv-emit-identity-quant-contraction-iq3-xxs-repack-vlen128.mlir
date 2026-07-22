// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-4 iq3_xxs GRID CONSTRUCTION -- the DUAL-ENTRY grid sibling, the SIXTH row of the
// weft::GridDecodePlan registry and the first whose grid entry covers only HALF an
// activation group. Landed the same four ways C4a-2/C4a-3 landed iq1_s/iq1_m (registry row
// + front door + BOTH emitter leaves + byte-exact oracle), because a registry row the
// front door cannot construct and no emitter leaf can lower would ship a "verifier accepts
// but nobody can lower" inconsistency. THIS test is the front-door half of that proof.
//
// The abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction op (iq3_xxs grid / decode)
// is AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler
// selection reads the STRUCTURED opponent facts (block_dot_compute_heavy = true) + the
// derived capability VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off
// the committed iq3_xxs dual-entry scale_model WHAT, so lowerToRepackGemvGrid CONSTRUCTS
// the typed weft_rvv.typed_repack_gemv_loop_body REGION (fold_model
// "grid_sign_dual_entry_single_scale_quarter") carrying the SINGLE
// weft_rvv.repack_gemv_grid_core integer-core brick (decode_model "iq3_xxs"),
// reconstructing the block_iq3_xxsx16 x16 weight facts 1696/160/32/1184 + the plain
// block_q8_K activation facts 292/4 (NO bsums -- its single-accumulator fold has no delta
// term), half_lanes=8 => mf2, numHalves==2.
//
// The FIXED 256-entry uint32 iq3xxs_grid + the DERIVED signs64 +-1 plane the abstract
// request DOES NOT carry are both RECONSTRUCTED at emit as static const decls (a true
// CONSTRUCT-from-abstract).
//
// NO perf/e2e claim of any kind: C4a-4 is a CONSTRUCTION line and never touched a board.
// Numeric correctness is gated separately + byte-exactly by
// tools/oracle-repack/oracle_repack_iq3_xxs.cpp (ZERO-MODEL, x86-native).

module {
  weft.exec.kernel @ggml_repack_gemv_iq3_xxs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq3_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The ABSTRACT dual-entry grid request: PLAIN block_iq3_xxs (stride 98 = fp16 d +
        // 3*(QK_K/8) = 96 qs bytes, qs @2) x PLAIN block_q8_K (stride 292), qk 256,
        // scale_model = the iq3_xxs dual-entry WHAT, block_dot_compute_heavy = true (routes
        // REPACK). It carries NO grid and NO sign plane -- the compiler RECONSTRUCTS both.
        // The result is dead (the repacked lane-wise grid GEVM sinks through the output
        // pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq3_xxs", scale_model = "superblock-d.fp16-grid-sign-dual-entry-4bit-scale-nomin-quarter", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 98 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the dual-entry grid typed_repack GEVM
// region (fold_model grid_sign_dual_entry_single_scale_quarter) carrying the grid core
// brick (decode_model iq3_xxs), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "grid_sign_dual_entry_single_scale_quarter"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-sign-dual-entry-4bit-scale-nomin-quarter"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// The repacked stride is 1696, NOT iq2_xxs's 1184: the grid-index strip DOUBLES (512 B ->
// 1024 B) because a uint32 entry covers half a group, so each group needs TWO indices.
// CONSTRUCT-SAME: weight_block_stride = 1696
// NO bsums attr is stamped: iq3_xxs's fold is the single-accumulator SignScaleStore shape
// with no delta term, so nothing reads a bsums plane. (The loop-body verifier REJECTS the
// attr outside the two folds that read it, so a stamp here would not survive.)
// CONSTRUCT-NOT: activation_bsums_byte_offset
// The in-region block_index-tied grid integer-core BRICK, decode_model iq3_xxs + the
// RECONSTRUCTED grid / ls / sign byte offsets (the abstract request carried plain facts).
// Unlike the iq1 rows, weight_sign_byte_offset here addresses a REAL sign-selector strip.
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq3_xxs"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 1184

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq3_xxs GEVM body via the NEW dual-entry leaf.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq3_xxs_q8_K_kernel_ggml_repack_gemv_iq3_xxs_q8_K(
// The FIXED 256-entry grid decl (RECONSTRUCTED by the compiler), emitted from the SAME
// canonical kIQ3XXSGrid the iq3_xxs BLOCK-DOT path emits, so the two paths cannot disagree.
// It is a uint32_t array -- the ONLY registered row whose grid is not int64. THAT is the
// fact that forces the new leaf.
// CHECK: verbatim "static const uint32_t weft_iq3xxs_grid[256] = {0x04040404U
// The DERIVED signs64 +-1 plane. iq3_xxs REUSES iq2_xxs's -- ggml's
// ggml_vec_dot_iq3_xxs_q8_K reads `ksigns_iq2xs` by that name, the same table its iq2_xxs
// sibling reads, so this is the SAME plane and not a lookalike.
// CHECK: verbatim "static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1,
// The body's grid + sign table POINTERS are the registry row's gridArrayName /
// signArrayName (pure DATA), NOT hard-coded emitter strings -- and they must name the SAME
// tables the decls above declare. The trailing quote makes each match exact, so a _BOGUS
// suffix cannot satisfy it.
// CHECK: literal "weft_iq3xxs_grid"
// CHECK: literal "weft_iq2xxs_signs64"
// Per-group weight base vx + x*nb*1696 (block_iq3_xxsx16 stride 1696).
// CHECK: literal "1696"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory grid GATHER: a u8 index strip -> vzext -> vsll -> vluxei16.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign fold onto the grid byte (iq3's grid bytes reach 62 < 128, so grid*sign fits i8).
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// The i32 in-block dot + the SINGLE per-sub-block ls-scale vmacc.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// THE DUAL-ENTRY SIGNATURE, and the load-bearing CHECK of this file: byte offset 1176.
//
// It is the LAST grid-index strip offset the dual-entry nest reaches --
// gridIdxOffset + (ib*8 + grp*2 + e)*16 + h*8 at ib=7, grp=3, e=1, h=1
//   = 160 + (56 + 6 + 1)*16 + 8 = 160 + 1008 + 8 = 1176
// -- and the single-base nest CANNOT reach it: its stride is (ib*4 + grp)*16, topping out
// at 160 + 496 + 8 = 664. (664 itself appears in BOTH, so it proves nothing; 1176 is the
// discriminator.) The sign-selector strip starts at 1184, so 1176 cannot come from there
// either.
//
// VERIFIED BY INJECTION, not by reasoning: disabling the emitter's
// `entryWidth == I32x4` dispatch arm makes iq3_xxs fall onto the single-base iq2_xxs leaf,
// and this CHECK goes RED.
//
// WHAT WAS TRIED FIRST AND THROWN AWAY, because it is the failure this line keeps making:
// the obvious CHECK is "the grid gather shifts by 2 while the sign gather still shifts by
// 3, so both literals must appear". That reads well and is WORTHLESS -- `literal "2"`
// occurs 128 times in the MIS-LOWERED output too (it is a common offset/immediate), so the
// CHECK passes either way. Had it shipped with its explanatory comment, this file would
// carry a confident sentence about a CHECK that tests nothing. The shift claim is true of
// the EMITTER (see the leaf) but it is not TESTABLE THIS WAY.
// CHECK: literal "1176"
// The SignScaleStore end-of-block fold + the 0.25f STORE constant -- ggml's
// `*s = 0.25f * sumf`, NOT the iq2 rows' 0.125f. The constant rides the registry row's
// storeScaleLiteral as data.
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.25f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The store-side step comment must name the constant it is ACTUALLY attached to. This row
// stores 0.25f, so its marker is quarter_scale; reusing the iq2 rows' "eighth_scale" here
// would ship a comment asserting something false, which is the exact failure this family
// has already made twice. The marker is DERIVED from storeScaleLiteral by a CLOSED map, so
// the two cannot drift.
// CHECK-NOT: callee=eighth_scale

// The grid decode is a MEMORY gather, NOT a register vrgather; the block-as-lane repack
// erases the cross-lane reduction wall; the dual-entry fold has NO min.
// NOWALL-NOT: vrgather
