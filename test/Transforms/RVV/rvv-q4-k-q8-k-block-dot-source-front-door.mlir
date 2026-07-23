// Track B auto-lowering, the SUPER-BLOCK rung -- one step ABOVE the codebook rung
// (rvv-iq4-nl-q8-0-block-dot-source-front-door.mlir). The COMPILER auto-CONSTRUCTS
// the complete weft.exec.kernel + exact canonical problem + one RVV variant around the
// typed SUPER-BLOCK dual-accumulator loop body
// (weft_rvv.typed_super_block_block_dot_loop_body) DECOMPOSED over the 5 q4_K bricks
// from a marked ggml `ggml_vec_dot_q4_K_q8_K` OPERATOR-IDENTITY source, instead of a
// per-kernel hand-authored super-block block-dot emitter input (M-FLAT milestone-3:
// the front door no longer constructs the opaque monolith weft_rvv.q4_k_q8_k_block_dot
// op). q4_K is the most-used modern K-quant and the HARDEST
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
// target is the mf2-default form, byte-identical to the retired monolith emit
// (modulo the source-op provenance token).

// The auto-constructed attr-less super-block block-dot scaffold (no shape knob).
// RUN: weft-opt %s --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door | FileCheck %s --check-prefix=BODY --implicit-check-not="scalar_fallback" --implicit-check-not="weft.exec.dispatch"
//
// The EMITTED super-block dot core from the SAME auto-constructed attr-less op:
// q4_K is NOT in any schedule autotuner, so there is NO --weft-rvv-materialize-schedule
// stamp step (unlike q4_0/iq4_nl); the op lowers DIRECTLY at the q4_K emitter's
// default mf2 integer-core anchor. The lowered kernel is BYTE-IDENTICAL to the
// retired hand-authored q4_K monolith block-dot emit (modulo the source-op
// provenance token) -- so the scalar-oracle byte-exactness holds transitively (the
// shared CORE helpers are ssh-rvv-pinned to ggml's real ggml_vec_dot_q4_K_q8_K
// _generic fp32 order). This lowering is VLEN-independent (no flip), so VLEN128 and
// VLEN256 emit the SAME bytes: COVERAGE, not a capability flip.
// RUN: weft-opt %s --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
//
// FAIL-CLOSED (I7): a non-conforming operator-identity signature (the q8_K activation
// operand is an f32 memref, not the i8 memref the vec_dot identity requires) is
// REJECTED, not silently constructed.
// RUN: not weft-opt %S/Inputs/q4-k-q8-k-block-dot-source-wrong-signature.mlir --weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door 2>&1 | FileCheck %s --check-prefix=BADSIG

module attributes {weft_rvv.source_front_door = "ggml_q4_K_q8_K_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q4_K_q8_K_kernel"} {
  func.func @source_q4_K_q8_K_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}

// ===================== AUTO-CONSTRUCTED TYPED SUPER-BLOCK LOOP BODY ==========
// The marked operator-identity source becomes a weft.exec.kernel with the
// auto-built typed SUPER-BLOCK DUAL-accumulator loop body op
// (weft_rvv.typed_super_block_block_dot_loop_body) DECOMPOSED into the 5 q4_K
// bricks + the four-value ABI set + the exact canonical problem. NO per-kernel
// emitter authored this, and NO opaque monolith weft_rvv.q4_k_q8_k_block_dot op is
// constructed (the M-FLAT milestone-3 flip: the front door now builds typed
// pattern-library primitives). NO shape knob is stamped (no integer_core_lmul on
// the loop op or the scaled-dot brick; the q4_K Win-A knob stays dormant, so the
// emitter lowers at its default mf2 anchor).
// BODY: weft.exec.kernel @ggml_vec_dot_q4_K_q8_K_kernel
// BODY: weft.exec.quantized_block_dot_problem @canonical_problem
// BODY-SAME: qk = 256
// BODY-SAME: weight_encoding = "ggml_q4_k_q8_k_block_dot"
// BODY: weft.exec.variant @rvv_q4_K_q8_K_block_dot
// The ggml vec_dot ABI value set -- the EXACT 4-role list (n, s, vx, vy), NO dead
// aux8/scales/aux32 scratch parameters: the super-block scratch is emitter-owned in
// the loop form, so the vestigial brick scratch operand slots are wired to the
// weight base %vx (not minted as placeholder runtime ABI values).
// BODY: weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"}
// BODY: weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// BODY: weft_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The typed super-block DUAL-accumulator loop body op (the OUTER nb = n/QK_K loop):
// the bounded super-block facts + the two-level fold model, but NO integer_core_lmul
// (the dormant q4_K Win-A knob -> emitter default mf2), and NO opaque monolith op.
// BODY: weft_rvv.typed_super_block_block_dot_loop_body
// BODY-SAME: activation_block_stride = 292 : i64
// BODY-SAME: fold_model = "super_block_two_level_scale_min"
// BODY-SAME: kind = "typed_super_block_block_dot_loop_body"
// BODY-SAME: qk = 256 : i64
// BODY-SAME: weight_block_stride = 144 : i64
// BODY-NOT: integer_core_lmul
// BODY-NOT: weft_rvv.q4_k_q8_k_block_dot
// The DUAL-accumulator region entry args: (super_block_index, sums vector<f32 m2>,
// sumf scalar f32).
// BODY: ^bb0(%{{.*}}: index, %{{.*}}: !weft_rvv.vector<f32, "m2">, %{{.*}}: f32):
// The 5 decomposed q4_K bricks, each keyed off the loop induction variable (region
// arg 0), so the per-super-block addressing is base + ib*stride (operand-driven).
// BODY: weft_rvv.q4_k_nibble_unpack %{{.*}}, %{{.*}} block %{{.*}}
// BODY: weft_rvv.q4_k_scale_min_bit_dance %{{.*}}, %{{.*}} block %{{.*}}
// BODY: weft_rvv.q4_k_scaled_dot %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// BODY: weft_rvv.q4_k_min_term %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// BODY: weft_rvv.q4_k_sums_fold_scale_d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// The dual carried-out yield (sums vector + sumf scalar).
// BODY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}} : !weft_rvv.vector<f32, "m2">, f32

// =================== EMITTED super-block dot core (default mf2) ==============
// The default mf2 integer-core anchor carries through the lowering byte-identical
// to the hand-authored q4_K block-dot emitter: the int8_t aux8[256] + uint32_t
// utmp[4] scratch, the e8m2(32) nibble unpack, the STRUCTURED 6-bit scale/min
// bit-dance (NO raw strings), the per-sub-block i32 dot, the MIN term, the deferred
// fp32 fold (SEPARATE vfmul/vfadd, NEVER a fused vfmacc), and the SEQUENTIAL
// horizontal sum. The block facts (256/8/144/292/offsets) are op structure.
// EMIT: emitc.func @weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(
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
// EMIT-NOT: emitc.func @weft_emitc_source_q4_K_q8_K_block_dot

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml Q4_K x Q8_K super-block block-dot source front door failed
