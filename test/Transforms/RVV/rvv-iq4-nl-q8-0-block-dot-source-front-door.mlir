// Track B auto-lowering, the CODEBOOK rung -- one step ABOVE the nibble-unpack rung
// (rvv-q4-0-q8-0-block-dot-source-front-door.mlir). The COMPILER auto-CONSTRUCTS the
// complete tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// attr-less tcrv_rvv.iq4_nl_q8_0_block_dot op from a marked ggml
// `ggml_vec_dot_iq4_nl_q8_0` OPERATOR-IDENTITY source, instead of a per-kernel
// hand-authored codebook block-dot emitter input. The genuinely-new structure this
// rung opens -- the 16-entry NON-LINEAR int8 CODEBOOK gather (the nibble INDEXES
// ggml's kvalues_iq4nl[16] via vrgather, it does NOT decode arithmetically) -- is
// FIRST-CLASS STRUCTURE inside that op and its existing FP4 codebook emitter; the
// front door supplies the 16-entry codebook as the structural DenseI8ArrayAttr and
// does NOT hand-roll the gather as fragile vector ops.
//
// THE CAPABILITY FLIP IS REAL HERE -- the honest inversion vs the q4_0 sibling
// (whose anchor is m1 at EVERY Zvl128b tier, no VLEN-byte flip). The codebook gather
// must index ALL 16 table entries, so its legal integer-core anchor is the
// VLEN-capability fact "the strip VLMAX spans the 16-entry table-index range": at
// VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16, PRUNED); at VLEN256 mf2 ALSO
// reaches VLMAX 16 (the ggml `_vl256` shape) and wins the capability-blind cost tie
// on its lighter peak-live footprint. So the SAME attr-less op FLIPS m1@VLEN128 ->
// mf2@VLEN256 -- a byte-different emitted codebook core. This rung CLEARS the
// non-NULL bar the q4_0 rung could not. The flip is the EXISTING unified schedule
// autotuner's, already board-sealed in
// rvv-iq4-nl-q8-0-block-dot-autotuner-divergence.mlir: the front door constructs the
// ATTR-LESS op and DEFERS shape selection to the unmodified
// --tcrv-rvv-materialize-schedule pass, so the flip rides the existing gearbox
// byte-for-byte. The NEW content is the auto-CONSTRUCTION feeding that gearbox; the
// front door REPRODUCES the sealed flip, it does not invent one.

// The auto-constructed attr-less codebook block-dot scaffold (no shape knobs --
// shape + the VLEN anchor flip are the autotuner's job).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door | FileCheck %s --check-prefix=BODY
//
// The REAL capability FLIP, asserted on the autotuner's STAMPED shape from the SAME
// auto-constructed attr-less op: VLEN128 stamps the m1 anchor (the only legal one);
// VLEN256 FLIPS to the mf2 anchor (the ggml _vl256 shape becomes legal+wins). Same
// construction, divergent anchor -- the existing unified gearbox decides.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// The FLIP carried to the EMITTED intrinsics: VLEN128 emits the e8m1 codebook core
// (vrgather_vv_i8m1, the whole 16-lane table in one cover); VLEN256 emits the
// BYTE-DIFFERENT e8mf2 core (vrgather_vv_i8mf2, the ggml _vl256 shape). The codebook
// gather is present in BOTH -- it is op structure; only the LMUL anchor flips.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN256
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

// ===================== AUTO-CONSTRUCTED ATTR-LESS BLOCK-DOT BODY ============
// The marked operator-identity source becomes a tcrv.exec.kernel with the
// auto-built attr-less tcrv_rvv.iq4_nl_q8_0_block_dot op + the four-value ABI set +
// the dispatch/fallback scaffold. NO per-kernel emitter authored this -- and NO
// shape knob is stamped here (the op is attr-less, so the autotuner is free to
// select the capability shape AND the VLEN anchor flip). The 16-entry codebook is
// carried as a structural DenseI8ArrayAttr.
// BODY: tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel
// BODY: tcrv.exec.variant @rvv_iq4_nl_q8_0_block_dot
// The ggml vec_dot ABI value set (n, s, vx, vy).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"}
// BODY: tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"}
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m1"
// BODY-SAME: sew = 32
// The attr-less codebook block-dot op: the bounded WHAT (kind/scale_model/block
// facts + the 16-entry codebook), but NO integer_core_lmul / multi_block_factor /
// strip_elision / minimum_vlen (the autotuner's job, including the VLEN flip).
// BODY: tcrv_rvv.iq4_nl_q8_0_block_dot
// BODY-SAME: activation_block_stride = 34 : i64
// BODY-SAME: activation_high_byte_offset = 16 : i64
// BODY-SAME: codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>
// BODY-SAME: kind = "ggml_iq4_nl_q8_0_block_dot"
// BODY-SAME: qk = 32 : i64
// BODY-SAME: quant_byte_offset = 2 : i64
// BODY-SAME: scale_model = "dual-fp16-per-block-d_x.d_y"
// BODY-SAME: weight_block_stride = 18 : i64
// BODY-NOT: integer_core_lmul
// BODY-NOT: multi_block_factor
// BODY-NOT: minimum_vlen
// BODY-NOT: strip_elision
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_iq4_nl_q8_0_block_dot_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_iq4_nl_q8_0_block_dot
// BODY: tcrv.exec.fallback @rvv_iq4_nl_q8_0_block_dot_scalar_fallback

// ===================== STAMPED SHAPE (the gearbox SELECTION) ================
// VLEN128 (rv64gcv): the EXISTING unified autotuner selected the m1 anchor -- the
// ONLY legal one (mf2's strip VLMAX 8 < 16 cannot index the 16-entry codebook table,
// PRUNED). minimum_vlen = 128 is stamped (the verifier recomputes gather legality
// from it).
// STAMP-VLEN128: tcrv_rvv.iq4_nl_q8_0_block_dot
// STAMP-VLEN128-SAME: integer_core_lmul = "m1"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: strip_elision = "elided"
//
// VLEN256 (rv64gcv_zvl256b): the SAME auto-constructed attr-less op FLIPS to the mf2
// anchor. The REAL VLEN fact -- stamped as minimum_vlen = 256 -- makes mf2's strip
// VLMAX reach 16 (it can now index the whole codebook table), so mf2 (the ggml
// _vl256 shape) becomes LEGAL and wins on the lighter peak-live footprint. The anchor
// MOVES with VLEN -- the Win-A capability-driven lowering, NOT a structural NULL.
// STAMP-VLEN256: tcrv_rvv.iq4_nl_q8_0_block_dot
// STAMP-VLEN256-SAME: integer_core_lmul = "mf2"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

// =============================== EMIT VLEN128 (the m1 anchor) ================
// The compiler-selected m1 anchor carries through the lowering: the codebook table +
// nibble gather + strip vsetvl are spelled at e8m1 (VLMAX 16 = the whole 16-lane
// half-block in ONE cover). This is BYTE-IDENTICAL to the already-board-validated
// m1/elided pinned form (rvv-to-emitc-iq4-nl-q8-0-block-dot.mlir, ssh-rvv byte-exact
// vs ggml's real ggml_vec_dot_iq4_nl_q8_0), modulo the variant symbol name -- so the
// scalar-oracle logic holds transitively and VLEN128 needs ZERO new board work.
// EMIT-VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(
// EMIT-VLEN128: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN128: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN128: return
// The residual operator-identity source func lowers to NOTHING (the same as the
// MVP/dequant/q4_0 sibling front doors): exactly ONE emitc kernel (the
// auto-constructed one).
// EMIT-VLEN128-NOT: emitc.func @tcrv_emitc_source_iq4_nl_q8_0_block_dot

// ============ EMIT VLEN256 (the mf2 FLIP, byte-different) ====================
// The compiler-selected mf2 anchor emits a BYTE-DIFFERENT codebook core: the table
// load / nibble gather / strip vsetvl narrow to e8mf2 (a FULL mf2 register at
// VLEN256, VLMAX 16 = the ggml _vl256 shape). This is the NON-NULL proof: the two
// VLENs do NOT emit the same bytes -- a capability FACT changes the lowering.
// EMIT-VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(
// EMIT-VLEN256: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN256: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN256: return

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// BADSIG: ggml IQ4_NL x Q8_0 codebook block-dot source front door failed
