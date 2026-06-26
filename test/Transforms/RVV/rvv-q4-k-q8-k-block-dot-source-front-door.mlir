// Track B auto-lowering, the SUPER-BLOCK rung -- one step ABOVE the codebook rung
// (rvv-iq4-nl-q8-0-block-dot-source-front-door.mlir). The COMPILER auto-CONSTRUCTS
// the complete tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// attr-less tcrv_rvv.q4_k_q8_k_block_dot op from a marked ggml `ggml_vec_dot_q4_K_q8_K`
// OPERATOR-IDENTITY source, instead of a per-kernel hand-authored super-block
// block-dot emitter input. q4_K is the most-used modern K-quant and the HARDEST
// Track B rung so far: the weight is a 256-element SUPER-BLOCK (8 sub-blocks of 32),
// each carrying a 6-bit scale + a 6-bit min PACKED across 12 scale/min bytes, plus a
// super-block fp16 d (@0) and fp16 dmin (@2). The super-block loop, the STRUCTURED
// 6-bit scale/min bit-dance, the aux32 i32 accumulation, and the DEFERRED two-level
// fp32 fold PLUS the q4_K MIN term are FIRST-CLASS STRUCTURE inside that op and its
// existing q4_K emitter (RVVToEmitCKQuant.cpp); the front door does NOT hand-roll
// any of it. The "scales" are NOT a front-door DenseArray (unlike iq4_nl's 16-entry
// codebook): the 6-bit scale/min pack is bytes inside the weight super-block decoded
// by the op from weight_scales_byte_offset == 4 -- so the front door stamps that
// offset, NOT a scales attr the op does not have.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP (the q4_0 sibling's framing, the INVERSE
// of iq4_nl's). q4_K has NO VLEN128-vs-VLEN256 byte-flip, AND -- unlike q4_0/iq4_nl --
// q4_K is NOT in any schedule-descriptor autotuner. So this front door does NOT "ride
// the existing gearbox": there is no q4_K gearbox to ride. The constructed attr-less
// op lowers at the q4_K emitter's DEFAULT "mf2" integer-core anchor (vsetvl_e8m2(32)
// per nibble group), VLEN-independent. The op's optional integer_core_lmul knob ("m1"
// narrows the chain) IS the q4_K Win-A LMUL -- but it is a DORMANT, emitter-sealed
// knob: neither auto-selected by a gearbox nor VLEN-flipped here. The byte-exact
// target is the mf2-default attr-less form (rvv-to-emitc-q4-k-q8-k-block-dot.mlir).

// The auto-constructed attr-less super-block block-dot scaffold (no shape knob).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-k-q8-k-block-dot-source-front-door | FileCheck %s --check-prefix=BODY
//
// The EMITTED super-block dot core from the SAME auto-constructed attr-less op:
// q4_K is NOT in any schedule autotuner, so there is NO --tcrv-rvv-materialize-schedule
// stamp step (unlike q4_0/iq4_nl); the op lowers DIRECTLY at the q4_K emitter's
// default mf2 integer-core anchor. The lowered kernel is BYTE-IDENTICAL to the
// hand-authored q4_K block-dot emitter input (rvv-to-emitc-q4-k-q8-k-block-dot.mlir),
// modulo the kernel/variant symbol names -- so the scalar-oracle byte-exactness holds
// transitively (that emitter is ssh-rvv-pinned to ggml's real ggml_vec_dot_q4_K_q8_K
// _generic fp32 order). This lowering is VLEN-independent (no flip), so VLEN128 and
// VLEN256 emit the SAME bytes: COVERAGE, not a capability flip.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-k-q8-k-block-dot-source-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
//
// FAIL-CLOSED (I7): a non-conforming operator-identity signature (the q8_K activation
// operand is an f32 memref, not the i8 memref the vec_dot identity requires) is
// REJECTED, not silently constructed.
// RUN: not tcrv-opt %S/Inputs/q4-k-q8-k-block-dot-source-wrong-signature.mlir --tcrv-rvv-materialize-q4-k-q8-k-block-dot-source-front-door 2>&1 | FileCheck %s --check-prefix=BADSIG

module attributes {tcrv_rvv.source_front_door = "ggml_q4_K_q8_K_block_dot_source",
                   tcrv_rvv.source_kernel = "ggml_vec_dot_q4_K_q8_K_kernel"} {
  func.func @source_q4_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED ATTR-LESS BLOCK-DOT BODY ============
// The marked operator-identity source becomes a tcrv.exec.kernel with the
// auto-built attr-less tcrv_rvv.q4_k_q8_k_block_dot op + the four-value ABI set +
// the dispatch/fallback scaffold. NO per-kernel emitter authored this -- and NO
// shape knob is stamped here (the op is attr-less; the q4_K Win-A integer_core_lmul
// stays dormant, so the emitter lowers at its default mf2 anchor).
// BODY: tcrv.exec.kernel @ggml_vec_dot_q4_K_q8_K_kernel
// BODY: tcrv.exec.variant @rvv_q4_K_q8_K_block_dot
// The ggml vec_dot ABI value set (n, s, vx, vy).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"}
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The attr-less super-block block-dot op: the bounded WHAT (kind/scale_model + the
// q4_K super-block facts), but NO integer_core_lmul (the dormant q4_K Win-A knob).
// BODY: tcrv_rvv.q4_k_q8_k_block_dot
// BODY-SAME: activation_block_stride = 292 : i64
// BODY-SAME: activation_bsums_byte_offset = 260 : i64
// BODY-SAME: activation_d_byte_offset = 0 : i64
// BODY-SAME: activation_quant_byte_offset = 4 : i64
// BODY-SAME: kind = "ggml_q4_k_q8_k_block_dot"
// BODY-SAME: qk = 256 : i64
// BODY-SAME: scale_model = "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min"
// BODY-SAME: sub_block = 32 : i64
// BODY-SAME: weight_block_stride = 144 : i64
// BODY-SAME: weight_d_byte_offset = 0 : i64
// BODY-SAME: weight_dmin_byte_offset = 2 : i64
// BODY-SAME: weight_qs_byte_offset = 16 : i64
// BODY-SAME: weight_scales_byte_offset = 4 : i64
// BODY-NOT: integer_core_lmul
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_q4_K_q8_K_block_dot_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_q4_K_q8_K_block_dot
// BODY: tcrv.exec.fallback @rvv_q4_K_q8_K_block_dot_scalar_fallback

// =================== EMITTED super-block dot core (default mf2) ==============
// The default mf2 integer-core anchor carries through the lowering byte-identical
// to the hand-authored q4_K block-dot emitter: the int8_t aux8[256] + uint32_t
// utmp[4] scratch, the e8m2(32) nibble unpack, the STRUCTURED 6-bit scale/min
// bit-dance (NO raw strings), the per-sub-block i32 dot, the MIN term, the deferred
// fp32 fold (SEPARATE vfmul/vfadd, NEVER a fused vfmacc), and the SEQUENTIAL
// horizontal sum. The block facts (256/8/144/292/offsets) are op structure.
// EMIT: emitc.func @tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(
// EMIT: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// EMIT: %[[UTMP:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"uint32_t">>
// EMIT: call_opaque "__riscv_vsetvl_e8m2"
// EMIT: call_opaque "__riscv_vle8_v_u8m2"
// EMIT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT: call_opaque "__riscv_vse8_v_i8m2"
// The deferred fp32 fold uses SEPARATE vfmul/vfadd, NEVER a fused FMA.
// EMIT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT: call_opaque "__riscv_vfmul_vf_f32m2"
// EMIT: call_opaque "__riscv_vfadd_vv_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfmacc
// EMIT-NOT: call_opaque "__riscv_vfmadd
// The sequential horizontal sum (NOT a vfredusum).
// EMIT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// EMIT: return
// The residual operator-identity source func lowers to NOTHING (the same as the
// MVP/dequant/q4_0/iq4_nl sibling front doors): exactly ONE emitc kernel.
// EMIT-NOT: emitc.func @tcrv_emitc_source_q4_K_q8_K_block_dot

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml Q4_K x Q8_K super-block block-dot source front door failed
