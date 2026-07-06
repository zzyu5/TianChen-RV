// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// iq2_s anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the grid core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x (both
// bases fold off the WEIGHT base via the per-body memo), so a valid operand edit yields
// DIFFERENT C (not a rejection). (MGATE) dropping the grid core's block_index still
// PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the driver
// gate requires the brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/iq2_s_q8_k_grid_core %%vx, %%vy/iq2_s_q8_k_grid_core %%vx, %%vx/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/iq2_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/iq2_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// iq2_s super-block SCALAR-accumulator per-half-scale GRID emit (the flip lowering + the
// shared per-super-block grid body, SIGN-PLANE explicit-signs variant, PER-HALF explicit
// scale, the LAST iq2 variant). iq2_s is a super-block quant whose sub-block decode is a
// codebook GRID GATHER (decode_model=lookup, the 1024-entry uint64 iq2s_grid gathered via
// vluxei16_v_i64m1 indexed by the 10-bit `qs[l] | ((qh<<(8-2l))&0x300)` -- a single index
// byte + 2 qh-plane bits -- + the UNIVERSAL signs256 SIGN plane gathered via a SECOND
// vluxei16 keyed by the RAW sign byte read DIRECTLY from the sign region at qs+32), the
// grid/explicit-signs SIBLING of the iq2_xs GRID core. Its fold is a SINGLE per-super-block
// scalar `sumf += d*(float)bsum` with the trailing `*s = 0.125f*sumf`, where bsum (the
// per-sub-block per-HALF-scaled grid/sign integer dot) is a SCALAR integer state -- the SAME
// scalar-accumulator arity as iq1_s's "scalar_delta_grid", with a DISTINCT integer core (the
// EXPLICIT per-sub-block 4-bit scales[8] two-half split ls1/ls2 + the 1024-grid + signs256
// gather + the qh-plane injection) and the trailing 0.125f factor. So
// tcrv_rvv.typed_super_block_block_dot_loop_body with fold_model = "scalar_delta_grid"
// carries a SCALAR accumulator: the region entry arguments are just
// (super_block_index, sumf:f32) and the region is terminated by
// tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE. The
// body carries the NET-NEW iq2_s per-half-scale GRID INTEGER CORE
// (tcrv_rvv.iq2_s_q8_k_grid_core -- the explicit scale byte -> ls1/ls2 two-half split, the
// qh-plane byte, the per-half 2-index vluxei16_v_i64m1 grid gather + the SECOND signs256
// vluxei16 gather + the vmul-onto-grid sign fold + the signed widening grid dot + the bsum
// fold) with a per-super-block `block %super_block_index` operand, producing the ONE SCALAR
// integer state bsum. Like iq2_xs the brick carries NO integer_core_lmul gearbox (fixed
// 16-lane per-half shape: i64m1 gather + i8m1 view + i16m2 widen at all VLEN).
//
// THE FLIP (L3 coverage): the front door now constructs this typed body as the SOLE
// representation of the iq2_s vec_dot (the monolith op + emitter + verifier were
// retired the same action). The emit is byte-identical to the retired monolith
// (same grid/signs256 decls + per-super-block grid body helper, same facts, same order)
// modulo the source-op provenance token + the func name. Numerical bit-exact-vs-ggml is
// pending-hardware (ssh rvv), not tested here. This is additive: it never touches the
// q4_K/q5_K DUAL, the q6_K SINGLE-vector, the q2_K scalar, nor the
// iq1_s/iq1_m/iq3_xxs/iq2_xxs/iq2_xs grid paths (their fold_model / brick branches are
// unchanged -- zero regression).

module {
  tcrv.exec.kernel @iq2_s_super_block_grid_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @iq2_s_super_block_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq2_s_super_block_grid_core, sew = 32 : i64, source_kernel = "iq2_s_super_block_grid_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 82 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // iq2_s per-half-scale GRID INTEGER CORE (NET-NEW, decode_model=lookup, explicit-
          // signs): the per-sub-block explicit scale byte `sc = scales[ib32]` -> `ls1 =
          // 2*(sc&0xf)+1` / `ls2 = 2*(sc>>4)+1`, the 10-bit grid index `qs[l] |
          // ((qh<<(8-2l))&0x300)` (a single index byte + 2 qh-plane bits), the EXPLICIT sign
          // byte read DIRECTLY from the sign region at qs+32, the per-half 2 grid indices + 2
          // sign bytes packed as u16 byte-offsets, TWO vluxei16_v_i64m1 gathers over the
          // 1024-entry uint64 iq2s_grid + the UNIVERSAL signs256 explicit-sign-byte plane,
          // the vmul-onto-grid sign fold, the signed widening grid dot, and `bsum += sumi*ls`
          // per half, producing the ONE SCALAR integer state bsum. The
          // `block %super_block_index` operand makes the weight/activation bases
          // per-super-block (vx + ib*82, vy + ib*292).
          %bsum = tcrv_rvv.iq2_s_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq2_s_q8_k_grid_core", scale_model = "per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-signs-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 82 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, weight_signs_byte_offset = 34 : i64, weight_qh_byte_offset = 66 : i64, weight_scales_byte_offset = 74 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += d*(float)bsum (d = fp16(x.d @0)*
          // y.d @0) + the trailing *s = 0.125f*sumf are emitter-inlined (no fold brick).
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block per-half-scale GRID body composes:
// the loop op (fold_model "scalar_delta_grid") + the iq2_s grid integer core brick + the
// one-operand scalar yield round-trip (the region carries the (index, f32) pair and the
// yield names the single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.iq2_s_q8_k_grid_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact
// SCALAR-accumulator per-half-scale GRID emit, byte-identical to the retired monolith (same
// grid/signs256 decls + per-super-block grid body helper, same facts, same order) modulo
// the source-op provenance token + the func name.
// EMIT: emitc.func @tcrv_emitc_iq2_s_super_block_grid_core_kernel_iq2_s_super_block_grid_core(
// The 1024-entry GRID codebook + the UNIVERSAL 2048-byte signs256 sign plane, emitted ONCE
// as structured static const decls (keyed off the grid-core brick op identity from the
// canonical kIQ2SGrid; signs256 = the definitional 8-bit-to-per-lane +-1 expansion, NO
// ksigns selector because iq2_s signs are EXPLICIT bytes).
// EMIT: verbatim "static const int64_t tcrv_iq2s_grid[1024] = {0x0808080808080808ULL,
// EMIT: verbatim "static const int8_t tcrv_iq2s_signs256[2048] = {1, 1, 1, 1, 1, 1, 1, 1, -1,
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the
// loop (NO 8-lane sums vector -- iq2_s's positive term is the scalar bsum).
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// The i64 grid64 + signs256 views set up ONCE above the loop.
// EMIT: callee=grid_table_i64_view
// EMIT: callee=signs_table_i64_view
// EMIT: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const int8_t">> to !emitc.ptr<!emitc.opaque<"const int64_t">>
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the grid
// core brick operands: vx + ib*82, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "82"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The per-super-block fp16*fp32 scale d (the fp16 weight d read seam + the fp32 y.d).
// EMIT: call_opaque "(float)*(const _Float16 *)"
// The ONE int32 SCALAR accumulator bsum (NO aux32 memory state, NO 8-lane vector).
// EMIT: local_variable=bsum
// EMIT: !emitc.lvalue<!emitc.opaque<"int32_t">>
// DELTA(c): the explicit per-sub-block scale byte -> ls1 = 2*(sc&0xf)+1, ls2 = 2*(sc>>4)+1.
// EMIT: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// EMIT: bitwise_right_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// DELTA(b): the 10-bit grid index = qs index byte OR ((qh<<(8-2l)) & 0x300), all in the INT
// domain (a single qs byte + 2 qh-plane bits; NO uint16 qs-word reassembly).
// EMIT: bitwise_left_shift %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// EMIT: bitwise_and %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// EMIT: bitwise_or %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// The HARDWARE vluxei16 grid+sign gather at the FIXED 16-lane per-half shape: the uint16_t
// gridoff/signoff[2] byte-offset arrays, vle16 the u16 indices (mf4 EMUL), TWO
// vluxei16_v_i64m1 gathers over grid64 + signs256, each reinterpreted to i8m1, the
// vmul-onto-grid sign fold, signed widening product, ONE vwredsum per HALF, extract. NO
// scalar vmerge/vneg/vand/vmsne (the sign comes from the signs256 gather, not a per-group
// scalar fold); NO i64m2 (the per-half scales ls1/ls2 force the 16-lane, not 32-lane,
// collapse -- so the shape is FIXED at m1, no VLEN gearbox).
// EMIT: "emitc.variable"() {{.*}} -> !emitc.array<2x!emitc.opaque<"uint16_t">>
// EMIT: call_opaque "__riscv_vle16_v_u16mf4"
// EMIT: call_opaque "__riscv_vluxei16_v_i64m1"
// EMIT: call_opaque "__riscv_vreinterpret_v_i64m1_i8m1"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vmul_vv_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// EMIT-NOT: call_opaque "__riscv_vmerge_vvm_i8m1"
// EMIT-NOT: call_opaque "__riscv_vmsne_vx_u8m1_b8"
// EMIT-NOT: call_opaque "__riscv_vluxei16_v_i64m2"
// The per-half bsum accumulate `bsum += sumi * ls` (integer, order-free).
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fp32 fold `sumf += d*(float)bsum` as ONE emitc.expression. NO
// 8-lane vfcvt/vfmul, NO post-loop vse32 horizontal sum.
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The trailing 0.125f factor + the *s store (`*s = 0.125f * sumf`, a SEPARATE statement
// OUTSIDE the loop; the store_s step comment precedes the 0.125f literal).
// EMIT: callee=store_s
// EMIT: literal "0.125f"

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the grid core's q8 base
// swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation
// arithmetic collapses (both bases fold to super_block_base_x off the weight base via
// the per-body memo), so no super_block_base_y appears between super_block_base_x and
// the fp16 scale read. This would be FALSE for the honestly-wired body (which emits
// super_block_base_y right there), so the emit provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=fcvt.s.h

// MGATE anti-bypass (the operand-driven driver gate): dropping the grid core's
// block_index parses + verifies clean (the standalone 4-operand form) but FAILS to
// legalize -- the driver requires the addressing brick's block_index to be the loop
// induction variable (region arg 0), so a body that would silently address
// super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'tcrv.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block GRID loop region
// is fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the
// offending op) -- the iq2_s grid core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7). The
// SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed: the
// scalar "scalar_delta_grid" path rejects an extra region arg (a vector-carrying
// dual/triple region) and a dual (second-operand) yield; the dual
// "super_block_two_level_scale_min" path rejects this scalar (index, sumf) region.
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator iq1_s fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// SCALARYIELDSECOND: must NOT carry a second
