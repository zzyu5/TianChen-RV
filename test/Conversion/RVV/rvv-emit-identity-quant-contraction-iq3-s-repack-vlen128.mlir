// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-5 iq3_s GRID CONSTRUCTION -- the SEVENTH and LAST row of the weft::GridDecodePlan
// registry, landed the same four ways C4a-2/C4a-3/C4a-4 landed iq1_s/iq1_m/iq3_xxs
// (registry row + front door + BOTH emitter leaves + byte-exact oracle). THIS test is the
// front-door half of that proof.
//
// WHAT IS DIFFERENT ABOUT THIS ROW, and therefore what this file is really pinning: iq3_s
// is the first row that added NO new axis value to the registry enums. Every axis it needs
// already existed, put there by a different sibling -- GridEntryWidth::I32x4 (iq3_xxs,
// which also means it REUSES that row's dual-entry emitter leaf unchanged in shape),
// GridSignPlane::Signs256 (iq2_s), a 512 entry count (iq2_xs), Single ls, SignScaleStore.
// So the CHECKs below are chosen to pin the two places where a row this close to iq3_xxs
// could be silently WRONG rather than to re-prove the shared skeleton:
//   (1) that it took the DUAL-ENTRY nest (the 2192 discriminator below), and
//   (2) that its distinct DATA -- the 512-entry grid, the signs256 plane, the u16 index
//       strip, the 2720 stride, the unit store -- is what actually reached the emit.
//
// The abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction op (iq3_s grid / decode)
// is AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler
// selection reads the STRUCTURED opponent facts (block_dot_compute_heavy = true) + the
// derived capability VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off
// the committed iq3_s explicit-sign dual-entry scale_model WHAT, so lowerToRepackGemvGrid
// CONSTRUCTS the typed weft_rvv.typed_repack_gemv_loop_body REGION (fold_model
// "grid_sign_dual_entry_single_scale_unit") carrying the SINGLE
// weft_rvv.repack_gemv_grid_core integer-core brick (decode_model "iq3_s"), reconstructing
// the block_iq3_sx16 x16 weight facts 2720/160/32/2208 + the plain block_q8_K activation
// facts 292/4 (NO bsums -- its single-accumulator fold has no delta term), half_lanes=8 =>
// mf2, numHalves==2.
//
// The FIXED 512-entry uint32 iq3s_grid + the DERIVED signs256 +-1 plane the abstract
// request DOES NOT carry are both RECONSTRUCTED at emit as static const decls (a true
// CONSTRUCT-from-abstract).
//
// NO perf/e2e claim of any kind: C4a-5 is a CONSTRUCTION line and never touched a board.
// Numeric correctness is gated separately + byte-exactly by
// tools/oracle-repack/oracle_repack_iq3_s.cpp (ZERO-MODEL, x86-native).

module {
  weft.exec.kernel @ggml_repack_gemv_iq3_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq3_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The ABSTRACT dual-entry explicit-sign grid request: PLAIN block_iq3_s (stride 110
        // = fp16 d + qs[64] + qh[8] + signs[32] + scales[4], qs @2) x PLAIN block_q8_K
        // (stride 292), qk 256, scale_model = the iq3_s WHAT, block_dot_compute_heavy = true
        // (routes REPACK). It carries NO grid and NO sign plane -- the compiler
        // RECONSTRUCTS both. The result is dead (the repacked lane-wise grid GEVM sinks
        // through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq3_s", scale_model = "superblock-d.fp16-grid-explicitsign-dual-entry-4bit-scale-nomin-unit", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the dual-entry grid typed_repack GEVM
// region (fold_model grid_sign_dual_entry_single_scale_unit) carrying the grid core brick
// (decode_model iq3_s), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "grid_sign_dual_entry_single_scale_unit"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-explicitsign-dual-entry-4bit-scale-nomin-unit"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// The repacked stride is 2720, NOT iq3_xxs's 1696: SAME dual-entry index count (8 per
// sub-block), but each index is a u16 rather than a u8, because iq3_s's index is 9 bits
// (`qs[e] | (((qh[ib] >> e) & 1) << 8)`, 0..511) and 511 does not fit a byte. That is the
// ONLY layout difference between the two rows: 1696 + 1024 = 2720.
// CONSTRUCT-SAME: weight_block_stride = 2720
// NO bsums attr is stamped: iq3_s's fold is the single-accumulator SignScaleStore shape
// with no delta term, so nothing reads a bsums plane. (The loop-body verifier REJECTS the
// attr outside the two folds that read it, so a stamp here would not survive.)
// CONSTRUCT-NOT: activation_bsums_byte_offset
// The in-region block_index-tied grid integer-core BRICK, decode_model iq3_s + the
// RECONSTRUCTED grid / ls / sign byte offsets (the abstract request carried plain facts).
// weight_sign_byte_offset addresses a REAL EXPLICIT sign strip (block_iq3_s carries a
// literal `uint8_t signs[QK_K/8]` plane), not a delta slot and not a ksigns selector.
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq3_s"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 2208

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq3_s GEVM body via the SHARED dual-entry leaf --
// the same leaf iq3_xxs uses, selected off the same entryWidth == I32x4 key.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq3_s_q8_K_kernel_ggml_repack_gemv_iq3_s_q8_K(
// The FIXED 512-entry grid decl (RECONSTRUCTED by the compiler), emitted from the SAME
// canonical kIQ3SGrid the iq3_s BLOCK-DOT path emits, so the two paths cannot disagree.
// 512 entries, not iq3_xxs's 256 -- and still a uint32_t array (the entry width is what the
// two iq3 rows SHARE; the count is what they do not).
// CHECK: verbatim "static const uint32_t weft_iq3s_grid[512] = {0x01010101U
// The DERIVED signs256 +-1 plane. This is the axis where iq3_s breaks with its own xxs
// sibling and sides with iq2_s: block_iq3_s carries EXPLICIT sign bytes tested by
// kmask_iq2xs, with NO ksigns_iq2xs selector indirection, so the plane must be indexed by
// all 256 byte values (2048 B) rather than 128 selectors (1024 B). It names iq2_s's table
// because the plane is a pure function of (byte, j) -- no format in it.
// CHECK: verbatim "static const int8_t weft_iq2s_signs256[2048] = {1, 1, 1, 1, 1, 1, 1, 1,
// A signs64 decl must NOT appear: emitting iq3_xxs's 1024-byte selector plane for this row
// would be a real mis-decode (the sign byte would be truncated to 7 bits), not a cosmetic
// slip.
// CHECK-NOT: weft_iq2xxs_signs64
// The body's grid + sign table POINTERS are the registry row's gridArrayName /
// signArrayName (pure DATA), NOT hard-coded emitter strings -- and they must name the SAME
// tables the decls above declare. The trailing quote makes each match exact, so a _BOGUS
// suffix cannot satisfy it.
// CHECK: literal "weft_iq3s_grid"
// CHECK: literal "weft_iq2s_signs256"
// Per-group weight base vx + x*nb*2720 (block_iq3_sx16 stride 2720).
// CHECK: literal "2720"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory grid GATHER. NOTE the vle16: iq3_s's 9-bit index strip is loaded
// DIRECTLY as a u16 lane, with NO vzext -- unlike iq3_xxs, whose u8 index strip needs the
// widen. This is the leaf's one PARAMETER (derived from the 512 entry count), and it is
// also why the sign gather below still needs vzext: the SIGN strip is a byte on both rows.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign fold onto the grid byte. iq3_s's grid bytes are all in [1,15] (machine-checked
// against ggml's iq3s_grid when this row landed: min 1, max 15, all odd), far under 128, so
// grid*sign fits i8 exactly as iq3_xxs's <= 62 do.
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// The i32 in-block dot + the SINGLE per-sub-block ls-scale vmacc. SINGLE, not dual: ggml
// reads ls1/ls2 from the two nibbles of scales[ib32/2] but steps ib32 += 2 and spends one
// on EACH 32-element sub-block, so a sub-block has exactly one ls and the repack lands them
// in the ordinary ls[8][16] strip.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// THE DUAL-ENTRY SIGNATURE, and the load-bearing CHECK of this file: byte offset 2192.
//
// It is the LAST grid-index strip offset the dual-entry nest reaches --
// gridIdxOffset + ((ib*8 + grp*2 + e)*16 + h*8)*2 at ib=7, grp=3, e=1, h=1
//   = 160 + (1008 + 8)*2 = 160 + 2032 = 2192
// -- and the single-base nest CANNOT reach it: its stride is ((ib*4 + grp)*16 + h*8)*2,
// topping out at 160 + (496+8)*2 = 1168. The sign strip starts at 2208, so 2192 cannot come
// from there either, and the whole gidx region ends at 2207.
//
// This is the SAME discriminator the iq3_xxs sibling file pins at 1176, moved by this row's
// u16 index strip. It is inherited deliberately rather than re-derived: the mis-lowering it
// guards against (falling onto the single-base iq2_xxs leaf, which reads the WRONG grid
// bytes for lanes 4-7 and returns confident garbage rather than failing) is identical for
// both I32x4 rows, and C4a-5 made that trap WIDER by adding a second row to it.
//
// VERIFIED BY INJECTION, not by reasoning: disabling the emitter's `entryWidth == I32x4`
// dispatch arm drops iq3_s onto the single-base iq2_xxs leaf, and the emit STILL SUCCEEDS
// (exit 0) -- it does not fail, it silently emits a kernel that reads the wrong grid bytes
// for lanes 4-7. `literal "2192"` goes to ZERO occurrences and this CHECK goes RED.
//
// Reported precisely rather than flatteringly: under THAT injection this is NOT the only
// CHECK that fires. The single-base leaf also emits an int64 grid decl, so the
// weft_iq3s_grid[512] CHECK above reddens FIRST, and vle16 -> 0 / vle8 -> 256 as the u16
// index strip collapses to a byte load. So the file catches that particular mis-lowering
// several ways over. What 2192 uniquely pins is the nest's REACH -- a mis-lowering that
// kept the right tables and the right strip width but hoisted one base per group would show
// up here and nowhere else above.
//
// Inherited warning from the iq3_xxs sibling, which applies verbatim: do NOT "improve" this
// into a CHECK on the gather shift literals ("grid shifts by 2, sign shifts by 3, so both
// must appear"). It reads well and tests NOTHING -- those literals occur throughout the
// mis-lowered output too. The shift claim is true of the emitter but is not testable this
// way.
// CHECK: literal "2192"
// The SignScaleStore end-of-block fold + the STORE constant. ggml's iq3_s ends `*s = sumf`
// -- a THIRD distinct store constant in this family (1, vs iq3_xxs's 0.25f and the iq2
// rows' 0.125f), and the one that cost the fold enum nothing because C4a-4 had already made
// the constant DATA. It rides the registry row's storeScaleLiteral as the literal 1.0f; see
// GridDecodePlan::storeScaleLiteral for why it is written as a multiply by one rather than
// as an absent field (short version: the empty literal is what a FORGOTTEN field looks like,
// and the leaves' refusal to emit on empty is a real gate worth more than one instruction).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "1.0f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The store-side step comment must name the constant it is ACTUALLY attached to. This row's
// constant is 1, so its marker is unit_scale; reusing quarter_scale or eighth_scale here
// would ship a comment asserting something false, which is the exact failure this family has
// already made twice. The marker is DERIVED from storeScaleLiteral by a CLOSED map, so the
// two cannot drift.
// CHECK-NOT: callee=eighth_scale
// CHECK-NOT: callee=quarter_scale

// The grid decode is a MEMORY gather, NOT a register vrgather; the block-as-lane repack
// erases the cross-lane reduction wall; the dual-entry fold has NO min.
// NOWALL-NOT: vrgather
