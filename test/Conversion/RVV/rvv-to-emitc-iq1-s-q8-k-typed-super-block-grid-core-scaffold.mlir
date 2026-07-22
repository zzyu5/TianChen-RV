// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// M-FLAT iq1_s milestone-2 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not
// gate-only): (MSWAP) swapping the grid core's q8 activation base %vy -> %vx CHANGES
// the emit -- the super_block_base_y activation-base arithmetic collapses into
// super_block_base_x (both bases fold off the WEIGHT base via the per-body memo), so
// a valid operand edit yields DIFFERENT C (not a rejection). (MGATE) dropping the
// grid core's block_index still PARSES + VERIFIES (the standalone 4-operand form) but
// FAILS to legalize -- the driver gate requires the brick's block_index to be the
// loop induction variable so the emit addresses base + ib*stride, never super-block-0.
// RUN: sed 's/iq1_s_q8_k_grid_core %%vx, %%vy/iq1_s_q8_k_grid_core %%vx, %%vx/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/iq1_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/iq1_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// M-FLAT iq1_s super-block SCALAR-accumulator GRID emit (milestone-2: byte-exact
// grid-gather lowering + the shared per-super-block grid body). iq1_s is a
// super-block quant whose sub-block decode is a codebook GRID GATHER
// (decode_model=lookup, the 2048-entry ternary iq1s_grid + vluxei16 gather), the
// super-block-flavor sibling of the FLAT codebook iq4_nl. Its fold is a SINGLE
// per-super-block scalar `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)`, where
// sumi (the qh-scaled ternary-grid positive dot) and sumi1 (the delta-bsum integer
// sum) are SCALAR integer states -- the SAME scalar-accumulator arity as q2_K's
// "scalar_scale_min", but a DISTINCT fold arithmetic and a GRID (not arithmetic)
// integer core. So weft_rvv.typed_super_block_block_dot_loop_body with fold_model =
// "scalar_delta_grid" carries a SCALAR accumulator: the region entry arguments are
// just (super_block_index, sumf:f32) and the region is terminated by
// weft_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE. The
// body carries the NET-NEW iq1_s TERNARY-grid INTEGER CORE
// (weft_rvv.iq1_s_q8_k_grid_core -- the 11-bit grid index build from qs+qh, the
// vluxei16 ternary-grid gather, the signed widening grid dot, the qh-encoded
// per-sub-block scale + delta sign, the delta-bsum sum) with a per-super-block
// `block %super_block_index` operand, producing the two SCALAR integer states
// sumi + sumi1.
//
// milestone-2 lowers the honest body to a REAL emitc.func: the byte-exact grid emit,
// byte-identical to the (not-yet-retired) monolith weft_rvv.iq1_s_q8_k_block_dot
// (same grid decl + per-super-block grid body helper, same facts, same order) modulo
// the source-op provenance token + the func name. This is NOT a flip -- the monolith
// remains the front-door route (six-state dispatch-wired); milestone-3 retires it and
// wires the front-door here. Numerical bit-exact-vs-ggml is pending-hardware (ssh
// rvv), not tested here. This is additive: it never touches the q4_K/q5_K DUAL, the
// q6_K SINGLE-vector, nor the q2_K scalar path (their fold_model branches are
// unchanged -- zero regression).

module {
  weft.exec.kernel @iq1_s_super_block_grid_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @iq1_s_super_block_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // iq1_s TERNARY-grid INTEGER CORE (NET-NEW, decode_model=lookup): the
          // 11-bit grid index build `qs[ib*4+l] | (((qh[ib] >> 3*l) & 7) << 8)`
          // + the vluxei16 gather over the 2048-entry ternary iq1s_grid + the
          // signed widening grid dot + the qh-encoded per-sub-block scale
          // `ls = 2*((qh[ib]>>12)&7)+1` + the delta sign `1 - 2*((qh[ib]>>15)&1)`
          // + the delta-bsum sum, producing the two SCALAR integer states sumi +
          // sumi1. The `block %super_block_index` operand makes the weight/
          // activation bases per-super-block (vx + ib*50, vy + ib*292).
          %sumi, %sumi1 = weft_rvv.iq1_s_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq1_s_q8_k_grid_core", scale_model = "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 2 : i64, weight_qh_byte_offset = 34 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64, groups_per_sub = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32, i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += d*((float)sumi +
          // IQ1S_DELTA*(float)sumi1) (d = fp16(x.d @0)*y.d @0, IQ1S_DELTA=0.125f)
          // is emitter-inlined (no separate fold brick).
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block GRID body composes: the
// loop op (fold_model "scalar_delta_grid") + the ternary-grid integer core brick +
// the one-operand scalar yield round-trip (the region carries the (index, f32)
// pair and the yield names the single carried-out `sumf` scalar, NO 8-lane `sums`
// vector).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: weft_rvv.iq1_s_q8_k_grid_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// Milestone-2: the honest body lowers to a REAL emitc.func -- the byte-exact
// SCALAR-accumulator GRID emit, byte-identical to the (not-yet-retired) monolith
// emitIQ1SQ8KBlockDot (same grid decl + per-super-block grid body helper, same
// facts, same order) modulo the source-op provenance token + the func name.
// EMIT: emitc.func @weft_emitc_iq1_s_super_block_grid_core_kernel_iq1_s_super_block_grid_core(
// The 2048-entry TERNARY GRID codebook emitted ONCE as a structured static const
// decl (keyed off the grid-core brick op identity from the canonical kIQ1SGrid).
// EMIT: verbatim "static const uint64_t weft_iq1s_grid[2048] = {0xffffffffffffffffULL,
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the
// loop (NO 8-lane sums vector -- iq1_s's positive term is the scalar sumi).
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the
// grid core brick operands: vx + ib*50, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "50"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The per-super-block fp16*fp32 scale d (the fp16 weight d read seam + the fp32 y.d).
// EMIT: call_opaque "(float)*(const _Float16 *)"
// The TWO int32 SCALAR accumulators sumi + sumi1 (NO aux32 memory state).
// EMIT: !emitc.lvalue<!emitc.opaque<"int32_t">>
// The HARDWARE vluxei16 ternary-grid gather: vle16 the u16 byte-offset indices (mf2
// EMUL), gather 4 grid u64 entries via vluxei16 over (const int64_t *)grid,
// reinterpret to 32 signed i8 grid bytes, signed widening product, ONE vwredsum,
// extract.
// EMIT: call_opaque "__riscv_vle16_v_u16mf2"
// EMIT: call_opaque "__riscv_vluxei16_v_i64m2"
// EMIT: call_opaque "__riscv_vreinterpret_v_i64m2_i8m2"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m4"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The DELTA term: the int16 bsums load folded by ls*delta.
// EMIT: load %{{.*}} : <!emitc.opaque<"const int16_t">>
// The per-super-block scalar delta fold `sumf += d*((float)sumi + 0.125f*
// (float)sumi1)` as ONE emitc.expression with the 0.125f (IQ1S_DELTA) literal. NO
// 8-lane vfcvt/vfmul, NO post-loop vse32 horizontal sum.
// EMIT: literal "0.125f"
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The *s store (NO trailing factor -- *s = sumf).
// EMIT: callee=store_s

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the grid core's q8 base
// swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation
// arithmetic collapses (both bases fold to super_block_base_x off the weight base
// via the per-body memo), so no super_block_base_y appears between super_block_base_x
// and the fp16 scale read. This would be FALSE for the honestly-wired body (which
// emits super_block_base_y right there), so the emit provably tracks the q8_base
// OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=fcvt.s.h

// MGATE anti-bypass (the operand-driven driver gate): dropping the grid core's
// block_index parses + verifies clean (the standalone 4-operand form) but FAILS to
// legalize -- the driver requires the addressing brick's block_index to be the loop
// induction variable (region arg 0), so a body that would silently address
// super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block GRID loop
// region is fail-closed rejected (the recursive [L-8] allowlist default is deny,
// naming the offending op) -- the iq1_s grid core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7).
// The SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed:
// the scalar "scalar_delta_grid" path rejects an extra region arg (a vector-carrying
// dual/triple region) and a dual (second-operand) yield; the dual
// "super_block_two_level_scale_min" path rejects this scalar (index, sumf) region.
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator iq1_s fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// SCALARYIELDSECOND: must NOT carry a second
