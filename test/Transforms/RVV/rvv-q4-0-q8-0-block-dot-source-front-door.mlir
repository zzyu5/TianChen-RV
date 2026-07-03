// Track B auto-lowering, the NIBBLE-UNPACK rung -- one step ABOVE the q8_0-style
// dequant rung. The COMPILER auto-CONSTRUCTS the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold from a marked ggml `ggml_vec_dot_q4_0_q8_0`
// OPERATOR-IDENTITY source (the eight vec_dot ABI roles n/s/bs/vx/bx/vy/by/nrc),
// instead of a per-kernel hand-authored block-dot emitter input.
//
// M-FLAT step 6 (the q4_0 typed-loop FLIP, the sibling of q8_0's step 5b): the
// q4_0 front door no longer constructs the ONE monolith tcrv_rvv.q4_0_q8_0_block_dot
// op -- it constructs the COMPLETE per-block TYPED LOOP chain
// (tcrv_rvv.typed_flat_block_dot_loop_body region) whose per-block integer core is
// the asymmetric OFFSET-BINARY packed-i4 x i8 product (the q4_0 nibble core:
// offset-binary (nibble-8) decode = xor-0x88 + low/high sign-extend + the vwmul/
// vwmacc against the two q8 halves) and whose fp32 fold is q4_0's LEFT-ASSOCIATIVE
// `sumf + ((float)sumi * d_x) * d_y`. The nibble unpack + the asymmetric i4xi8
// product are FIRST-CLASS STRUCTURE (typed ops) inside that region -- the front
// door does NOT hand-roll them as fragile vector ops.
//
// The typed core runs the packed-i4 product/reduce at the SEW8/m1 HALF-block
// anchor (integer_core_lmul m1, multi_block_factor pinned to 1, strip_elision
// elided), so the setvl/with_vl config is sew=8/m1 (NOT the shared monolith
// sew=32/m1). The remaining monolith rows stay byte-unchanged.
//
// Byte-exactness to the hand-authored q4_0 emitter is pinned out-of-band: the
// typed-loop body lowers BYTE-IDENTICALLY to the monolithic q4_0 mbf1 body (the
// OffsetBinaryNibble / half-block / LeftAssoc instance of emitFlatBlockDot),
// asserted by the full-body lit (rvv-to-emitc-typed-flat-block-dot-loop-full-body)
// + the full-pipeline export e2e; the shape-adaptive capability FLIP for the
// monolith op stays exercised by rvv-q4-0-q8-0-block-dot-autotuner-divergence.mlir.
// Front-door shape-adaptivity (the multi_block unroll) is a later M-FLAT step.

// The auto-constructed typed flat block-dot loop chain (mbf1/m1/elided, no shape
// knobs -- shape-adaptivity is a later M-FLAT step).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=BODY
//
// The constructed loop chain lowers to the q4_0 nibble core (op structure), a
// light emit-presence check (byte-exactness is locked by the full-body /
// full-pipeline lits).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
//
// FAIL-CLOSED (I7): a non-conforming operator-identity signature (the q8 activation
// operand is an f32 memref, not the i8 memref the vec_dot identity requires) is
// REJECTED, not silently constructed.
// RUN: not tcrv-opt %S/Inputs/q4-0-q8-0-block-dot-source-wrong-signature.mlir --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door 2>&1 | FileCheck %s --check-prefix=BADSIG

module attributes {tcrv_rvv.source_front_door = "ggml_q4_0_q8_0_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel"} {
  func.func @source_q4_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED TYPED LOOP CHAIN ====================
// The marked operator-identity source becomes a tcrv.exec.kernel with the
// auto-built typed flat block-dot loop chain + the full ABI value set + the
// dispatch/fallback scaffold. NO per-kernel emitter authored this.
// BODY: tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel
// BODY: tcrv.exec.variant @rvv_q4_0_q8_0_block_dot
// The ggml vec_dot ABI value set (n, s, bs, vx, bx, vy, by, nrc).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"}
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// The typed core runs at the SEW8/m1 half-block anchor (NOT the monolith sew=32).
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 8
// The typed flat block-dot LOOP body op: q4_0's left-assoc fold, m1 integer core,
// half-block (qk/2) geometry (18/34 strides), mbf1 (no multi_block_factor attr).
// BODY: tcrv_rvv.typed_flat_block_dot_loop_body
// BODY-SAME: activation_block_stride = 34 : i64
// BODY-SAME: fold_model = "left_assoc"
// BODY-SAME: integer_core_lmul = "m1"
// BODY-SAME: kind = "typed_flat_block_dot_loop_body"
// BODY-SAME: qk = 32 : i64
// BODY-SAME: strip_elision = "elided"
// BODY-SAME: weight_block_stride = 18 : i64
// BODY-NOT: multi_block_factor
// The per-block integer core is the asymmetric offset-binary packed-i4 x i8
// product (i4m1 x i8m1x2 -> i16m2) reduced to the i32m1 lane -> scalar sumi.
// BODY: tcrv_rvv.packed_i4_offset_binary_x_i8_product
// BODY-SAME: kind = "signed_packed_i4_offset_binary_x_i8_product"
// BODY-SAME: product_relation = "offset-binary-i4m1-x-i8m1x2-to-i16m2"
// BODY: tcrv_rvv.standalone_reduce
// BODY: tcrv_rvv.typed_vector_lane0_to_scalar_extract
// The per-block dequant (brick 2) + cross-block f32 fold (brick 3) + the yield.
// BODY: tcrv_rvv.block_computed_scale_dequant
// BODY: tcrv_rvv.cross_block_f32_accumulate
// BODY: tcrv_rvv.typed_flat_block_dot_loop_yield
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_q4_0_q8_0_block_dot_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_q4_0_q8_0_block_dot
// BODY: tcrv.exec.fallback @rvv_q4_0_q8_0_block_dot_scalar_fallback

// ===================== LOWERED q4_0 NIBBLE CORE (op structure) ==============
// The typed loop lowers to the ONE auto-constructed emitc kernel whose integer
// core is the q4_0 nibble-unpack chain (vxor.vx 0x88 + vsll/vsra sign-extend +
// vwmul/vwmacc against the q8 halves) + the i16m2 -> i32m1 reduce. Byte-exactness
// is locked by the full-body / full-pipeline lits.
// EMIT: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vxor_vx_i8m1"
// EMIT: call_opaque "__riscv_vsll_vx_i8m1"
// EMIT: call_opaque "__riscv_vsra_vx_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwmacc_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: return
// The residual operator-identity source func lowers to NOTHING (the marker-only
// removeAttr): there is exactly ONE emitc kernel (the auto-constructed one).
// EMIT-NOT: emitc.func @tcrv_emitc_source_q4_0_q8_0_block_dot

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml Q4_0 x Q8_0 block-dot source front door failed
