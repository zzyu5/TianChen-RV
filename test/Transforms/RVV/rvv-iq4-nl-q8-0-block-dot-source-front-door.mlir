// Track B auto-lowering, the CODEBOOK rung -- one step ABOVE the nibble-unpack rung
// (rvv-q4-0-q8-0-block-dot-source-front-door.mlir). The COMPILER auto-CONSTRUCTS the
// complete tcrv.exec.kernel + variant + dispatch/fallback scaffold from a marked ggml
// `ggml_vec_dot_iq4_nl_q8_0` OPERATOR-IDENTITY source, instead of a per-kernel
// hand-authored codebook block-dot emitter input.
//
// L3-iq4_nl M2 (the iq4_nl typed-loop FLIP, the CODEBOOK sibling of q4_0's step 6):
// the iq4_nl front door no longer constructs the ONE monolith
// tcrv_rvv.iq4_nl_q8_0_block_dot op (RETIRED at M2) -- it constructs the COMPLETE
// per-block TYPED LOOP chain (tcrv_rvv.typed_flat_block_dot_loop_body region) whose
// integer core is the 2nd primitive class: a 16-entry NON-LINEAR int8 CODEBOOK table
// broadcast (tcrv_rvv.codebook_table_broadcast) + the asymmetric CODEBOOK-GATHER
// packed-i4 x plain-i8 product (tcrv_rvv.codebook_gather_x_i8_product: the nibble
// INDEXES ggml's kvalues_iq4nl[16] via vrgather, it does NOT decode arithmetically),
// and whose fp32 fold is iq4_nl's scales-first `sumf + (float)sumi * (d_x*d_y)`
// (fold_model "sumi_times_scales", node-identical to q8_0). The codebook table +
// gather are FIRST-CLASS STRUCTURE (typed ops) inside that region -- the front door
// does NOT hand-roll the gather as fragile vector ops, and supplies the 16-entry
// codebook as the structural DenseI8ArrayAttr on the broadcast brick.
//
// Unlike the SEW8 plain flat cores, the iq4_nl codebook core frames its OUTER
// setvl/with_vl at sew=32/m1 (the standalone_reduce codebook framing; the e8m1
// gather runs its own vsetvl INSIDE the region). integer_core_lmul m1, mbf1 (no
// multi_block_factor), strip_elision elided. The remaining monolith rows stay
// byte-unchanged.
//
// Byte-exactness to the hand-authored iq4_nl emitter is pinned out-of-band: at
// milestone M1 (commit 197a7a5b, before the monolith was retired) the typed-loop body
// was proven byte-IDENTICAL to the monolithic iq4_nl mbf1/elided m1 body (the
// CodebookGatherNibble / half-block / SumiTimesScales instance of emitFlatBlockDot) by
// an EMPTY canonicalized region-vs-monolith diff on a forced clean rebuild; the codebook
// emit is asserted directly by rvv-to-emitc-iq4-nl-q8-0-typed-flat-block-dot-loop-body.mlir
// + the full-pipeline export e2e.

// The auto-constructed typed flat block-dot loop chain (codebook branch; mbf1/m1/
// elided, no shape knobs).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=BODY
//
// The constructed loop chain lowers to the iq4_nl codebook core (op structure), a
// light emit-presence check (byte-exactness is locked by the full-body /
// full-pipeline lits).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
//
// FAIL-CLOSED (I7): a non-conforming operator-identity signature (the q8 activation
// operand is an f32 memref, not the i8 memref the vec_dot identity requires) is
// REJECTED, not silently constructed.
// RUN: not tcrv-opt %S/Inputs/iq4-nl-q8-0-block-dot-source-wrong-signature.mlir --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door 2>&1 | FileCheck %s --check-prefix=BADSIG

module attributes {tcrv_rvv.source_front_door = "ggml_iq4_nl_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel"} {
  func.func @source_iq4_nl_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED TYPED LOOP CHAIN ====================
// The marked operator-identity source becomes a tcrv.exec.kernel with the
// auto-built typed flat block-dot loop chain + the full ABI value set + the
// dispatch/fallback scaffold. NO per-kernel emitter authored this.
// BODY: tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel
// BODY: tcrv.exec.variant @rvv_iq4_nl_q8_0_block_dot
// The ggml vec_dot ABI value set (n, s, vx, vy).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"}
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// The codebook core frames its OUTER config at SEW32/m1 (NOT the SEW8 plain cores).
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The typed flat block-dot LOOP body op: iq4_nl's scales-first fold, m1 integer core,
// half-block (qk/2) geometry (18/34 strides), mbf1 (no multi_block_factor attr).
// BODY: tcrv_rvv.typed_flat_block_dot_loop_body
// BODY-SAME: activation_block_stride = 34 : i64
// BODY-SAME: fold_model = "sumi_times_scales"
// BODY-SAME: integer_core_lmul = "m1"
// BODY-SAME: kind = "typed_flat_block_dot_loop_body"
// BODY-SAME: qk = 32 : i64
// BODY-SAME: strip_elision = "elided"
// BODY-SAME: weight_block_stride = 18 : i64
// BODY-NOT: multi_block_factor
// The per-block CODEBOOK integer core: the 16-entry non-linear int8 table broadcast
// (carried as a structural DenseI8ArrayAttr) + the asymmetric codebook-gather packed-i4
// x i8 product (i8m1 x i8m1x2 -> i16m2) reduced to the i32m1 lane -> scalar sumi.
// BODY: tcrv_rvv.codebook_table_broadcast
// BODY-SAME: codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>
// BODY-SAME: table_symbol = "tcrv_iq4_nl_kvalues"
// BODY: tcrv_rvv.codebook_gather_x_i8_product
// BODY-SAME: kind = "signed_codebook_gather_x_i8_product"
// BODY-SAME: product_relation = "codebook-gather-i8-x-i8x2-to-i16"
// BODY: tcrv_rvv.standalone_reduce
// BODY: tcrv_rvv.typed_vector_lane0_to_scalar_extract
// The per-block dequant (brick 2) + cross-block f32 fold (brick 3) + the yield.
// BODY: tcrv_rvv.block_computed_scale_dequant
// BODY: tcrv_rvv.cross_block_f32_accumulate
// BODY: tcrv_rvv.typed_flat_block_dot_loop_yield
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_iq4_nl_q8_0_block_dot_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_iq4_nl_q8_0_block_dot
// BODY: tcrv.exec.fallback @rvv_iq4_nl_q8_0_block_dot_scalar_fallback

// ===================== LOWERED iq4_nl CODEBOOK CORE (op structure) ==========
// The typed loop lowers to the ONE auto-constructed emitc kernel whose integer core
// is the codebook gather chain: the 16-entry table decl + broadcast, the nibble split
// (vand 0x0F / vsrl 0x04) into two UNSIGNED index lanes, the vrgather through the
// broadcast table, and the shared asymmetric vwmul/vwmacc -> i16m2 -> i32m1 reduce.
// The q4_0 offset-binary decode chain (vxor 0x88) is ABSENT. Byte-exactness is locked
// by the full-body / full-pipeline lits.
// EMIT: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(
// EMIT: verbatim "static const int8_t tcrv_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vle8_v_u8m1"
// EMIT: call_opaque "__riscv_vand_vx_u8m1"
// EMIT: call_opaque "__riscv_vsrl_vx_u8m1"
// EMIT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwmacc_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT-NOT: call_opaque "__riscv_vxor_vx_i8
// EMIT: return
// The residual operator-identity source func lowers to NOTHING (the marker-only
// removeAttr): there is exactly ONE emitc kernel (the auto-constructed one).
// EMIT-NOT: emitc.func @tcrv_emitc_source_iq4_nl_q8_0_block_dot

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml IQ4_NL x Q8_0 codebook block-dot source front door failed
