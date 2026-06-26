// Track B auto-lowering, the NIBBLE-UNPACK rung -- one step ABOVE the q8_0-style
// dequant rung (rvv-widening-dot-reduce-dequantize-source-front-door.mlir). The
// COMPILER auto-CONSTRUCTS the complete tcrv.exec.kernel + variant +
// dispatch/fallback scaffold around ONE attr-less tcrv_rvv.q4_0_q8_0_block_dot op
// from a marked ggml `ggml_vec_dot_q4_0_q8_0` OPERATOR-IDENTITY source (the eight
// vec_dot ABI roles n/s/bs/vx/bx/vy/by/nrc), instead of a per-kernel hand-authored
// block-dot emitter input. The NIBBLE UNPACK (offset-binary (nibble-8) decode =
// xor-0x88 + low/high sign-extend + the asymmetric i4xi8 widening product) is
// FIRST-CLASS STRUCTURE inside that op and its existing emitter -- the front door
// does NOT hand-roll it as fragile vector ops.
//
// CAPABILITY FLIP -- the HONEST framing. q4_0's integer-core anchor is m1 at EVERY
// Zvl128b tier, so there is NO VLEN128-vs-VLEN256 byte-flip (unlike the MVP/dequant
// LMUL flip). The REAL q4_0 divergence is Zvl128b-gated and lives in the EXISTING
// schedule autotuner. So the front door constructs the ATTR-LESS op and DEFERS
// shape selection to the unmodified --tcrv-rvv-materialize-q4-0-schedule pass: the
// constructed op is byte-identical to that pass's hand-authored input
// (rvv-q4-0-q8-0-block-dot-autotuner-divergence.mlir), so the capability flip
// rides the existing gearbox byte-for-byte. The NEW content is the
// auto-CONSTRUCTION feeding that gearbox, not a new flip.

// The auto-constructed attr-less block-dot scaffold (no shape knobs -- shape is
// the autotuner's job).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=BODY
//
// The capability FLIP, asserted on the autotuner's STAMPED shape from the SAME
// auto-constructed attr-less op: rv64gcv stamps (m1, factor=4, elided); the
// constrained rv64gc_zve32x prunes the elided shapes and stamps (m1, factor=2,
// robust). Same construction, divergent shape -- the existing gearbox decides.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q4-0-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-FULLV
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q4-0-schedule=march=rv64gc_zve32x | FileCheck %s --check-prefix=STAMP-ZVE32X
//
// The FLIP carried to the EMITTED intrinsics: full-V emits the strip-elided form
// (FOUR adjacent vwredsums, no inner strip for-loop); zve32x emits the robust
// strip-loop form. The nibble-unpack chain (vxor.vx 0x88 + vsll/vsra + vwmul/vwmacc)
// is present in BOTH -- it is op structure, capability-independent.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q4-0-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLV
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-q4-0-schedule=march=rv64gc_zve32x --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=ZVE32X
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

// ===================== AUTO-CONSTRUCTED ATTR-LESS BLOCK-DOT BODY ============
// The marked operator-identity source becomes a tcrv.exec.kernel with the
// auto-built attr-less tcrv_rvv.q4_0_q8_0_block_dot op + the full ABI value set +
// the dispatch/fallback scaffold. NO per-kernel emitter authored this -- and NO
// shape knob is stamped here (the op is attr-less, so the autotuner is free to
// select the capability shape).
// BODY: tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel
// BODY: tcrv.exec.variant @rvv_q4_0_q8_0_block_dot
// The ggml vec_dot ABI value set (n, s, bs, vx, bx, vy, by, nrc).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"}
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The attr-less block-dot op: the bounded WHAT (kind/scale_model/block facts), but
// NO integer_core_lmul / multi_block_factor / strip_elision (the autotuner's job).
// BODY: tcrv_rvv.q4_0_q8_0_block_dot
// BODY-SAME: activation_block_stride = 34 : i64
// BODY-SAME: activation_high_byte_offset = 16 : i64
// BODY-SAME: kind = "ggml_q4_0_q8_0_block_dot"
// BODY-SAME: qk = 32 : i64
// BODY-SAME: quant_byte_offset = 2 : i64
// BODY-SAME: scale_model = "dual-fp16-per-block-d_x.d_y"
// BODY-SAME: weight_block_stride = 18 : i64
// BODY-NOT: integer_core_lmul
// BODY-NOT: multi_block_factor
// BODY-NOT: strip_elision
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_q4_0_q8_0_block_dot_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_q4_0_q8_0_block_dot
// BODY: tcrv.exec.fallback @rvv_q4_0_q8_0_block_dot_scalar_fallback

// ===================== STAMPED SHAPE (the gearbox SELECTION) ================
// rv64gcv (Zvl128b): the EXISTING autotuner selected (m1, factor=4, elided) on the
// SAME auto-constructed attr-less op.
// STAMP-FULLV: tcrv_rvv.q4_0_q8_0_block_dot
// STAMP-FULLV-SAME: integer_core_lmul = "m1"
// STAMP-FULLV-SAME: multi_block_factor = 4 : i64
// STAMP-FULLV-SAME: strip_elision = "elided"
// STAMP-FULLV-SAME: tcrv_rvv.q4_0_schedule.has_zvl128b = true
//
// rv64gc_zve32x (no Zvl128b): the SAME op gets (m1, factor=2, robust) -- the elided
// shapes were pruned. The shape DIVERGES purely by the capability fact.
// STAMP-ZVE32X: tcrv_rvv.q4_0_q8_0_block_dot
// STAMP-ZVE32X-SAME: integer_core_lmul = "m1"
// STAMP-ZVE32X-SAME: multi_block_factor = 2 : i64
// STAMP-ZVE32X-SAME: strip_elision = "robust"
// STAMP-ZVE32X-SAME: tcrv_rvv.q4_0_schedule.has_zvl128b = false

// =============================== FULL-V (rv64gcv) ===========================
// The selected (m1, factor=4, elided) shape: FOUR adjacent elided integer cores
// (each ONE vsetvl_e8m1 + ONE vwredsum, NO inner strip for-loop). The nibble-unpack
// chain is op structure (present regardless of shape).
// FULLV: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(
// FULLV: call_opaque "__riscv_vxor_vx_i8m1"
// FULLV: call_opaque "__riscv_vsll_vx_i8m1"
// FULLV: call_opaque "__riscv_vsra_vx_i8m1"
// FULLV: call_opaque "__riscv_vwmul_vv_i16m2"
// FULLV: call_opaque "__riscv_vwmacc_vv_i16m2"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: return
// The residual operator-identity source func (left by the marker-only removeAttr,
// the same as the MVP/dequant sibling front doors) lowers to NOTHING: there is
// exactly ONE emitc kernel (the auto-constructed one). Byte-exactness to the
// hand-authored block-dot emitter is pinned out-of-band: this lowered kernel is
// byte-identical to rvv-q4-0-q8-0-block-dot-autotuner-divergence.mlir's full-V
// output (modulo the variant symbol name), whose integer core is ssh-rvv-pinned
// to ggml's real ggml_vec_dot_q4_0_q8_0 -> the scalar-oracle logic holds
// transitively.
// FULLV-NOT: emitc.func @tcrv_emitc_source_q4_0_q8_0_block_dot

// =============================== ZVE32X (no Zvl128b) ========================
// The selected (m1, factor=2, robust) shape: per-block ROBUST integer cores that
// KEEP the inner strip for-loop (VLEN-robust). Same nibble-unpack op structure.
// ZVE32X: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(
// ZVE32X: call_opaque "__riscv_vxor_vx_i8m1"
// ZVE32X: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// ZVE32X: return

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml Q4_0 x Q8_0 block-dot source front door failed
