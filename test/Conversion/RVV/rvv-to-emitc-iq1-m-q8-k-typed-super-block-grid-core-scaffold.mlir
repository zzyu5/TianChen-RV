// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// iq1_m anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the grid core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x
// (both bases fold off the WEIGHT base via the per-body memo), so a valid operand edit
// yields DIFFERENT C (not a rejection). (MGATE) dropping the grid core's block_index
// still PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the
// driver gate requires the brick's block_index to be the loop induction variable so
// the emit addresses base + ib*stride, never super-block-0.
// RUN: sed 's/iq1_m_q8_k_grid_core %%vx, %%vy/iq1_m_q8_k_grid_core %%vx, %%vx/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/iq1_m_q8_k_grid_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/iq1_m_q8_k_grid_core %%vx, %%vy, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND
// The brick verifier is fail-closed on the iq1_m format facts (I7): a wrong
// weight_block_stride (56 = block_iq1_m) is rejected.
// RUN: sed 's/weight_block_stride = 56 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset/weight_block_stride = 99 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADSTRIDE

// iq1_m super-block SCALAR-accumulator GRID emit (the flip lowering, iq1_s SIBLING).
// iq1_m is the second GRID/codebook super-block member -- it REUSES the whole iq1_s
// scalar-delta-grid scaffold (the SAME loop op fold_model "scalar_delta_grid", the SAME
// single `sumf` scalar accumulator + (index, sumf:f32) region contract, the SAME
// selector, the SAME single-yield), with a DISTINCT in-loop brick: the iq1_m
// TERNARY-grid INTEGER CORE (tcrv_rvv.iq1_m_q8_k_grid_core). iq1_m's integer decode
// differs from iq1_s: a packed iq1m_scale fp16 RECONSTRUCT from the 4 scales[] words,
// TWO per-sub-block half scales ls1/ls2, a HALF-SPLIT per-half vluxei16 grid dot (i8m1/
// i16m2 over 16 lanes), and a per-GROUP delta with FOUR independent signs reduced from
// a FRESH Σq8 (NO bsums). It produces the two SCALAR integer states sumi1 (grid) +
// sumi2 (delta). The `block %super_block_index` operand makes the weight/activation
// bases per-super-block (vx + ib*56, vy + ib*292).
//
// The honest body lowers to a REAL emitc.func: the byte-exact grid emit, byte-identical
// to the (now-retired) monolith tcrv_rvv.iq1_m_q8_k_block_dot (same grid decl + shared
// per-super-block grid body helper, same facts, same order) modulo the source-op
// provenance token + the func name. This is the C2 marginal-cost payoff: the SECOND
// grid member reuses the iq1_s scaffold and only adds a variant brick. Numerical
// bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. Additive -- it never
// touches the iq1_s, q4_K/q5_K, q6_K, nor q2_K fold_model branches (zero regression).

module {
  tcrv.exec.kernel @iq1_m_super_block_grid_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @iq1_m_super_block_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq1_m_super_block_grid_core, sew = 32 : i64, source_kernel = "iq1_m_super_block_grid_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 56 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // iq1_m TERNARY-grid INTEGER CORE (decode_model=lookup, iq1_s sibling): the
          // packed iq1m_scale fp16 reconstruct + the half-split per-half vluxei16 grid
          // dot with two half scales + the per-group four-sign Σq8 delta, producing the
          // two SCALAR integer states sumi1 + sumi2. The `block %super_block_index`
          // operand makes the bases per-super-block (vx + ib*56, vy + ib*292).
          %sumi1, %sumi2 = tcrv_rvv.iq1_m_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq1_m_q8_k_grid_core", scale_model = "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-delta-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 56 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 32 : i64, weight_scales_byte_offset = 48 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += d*((float)sumi1 +
          // IQ1M_DELTA*(float)sumi2) (d = reconstructed-fp16 * y.d @0, IQ1M_DELTA=0.125f)
          // is emitter-inlined (no separate fold brick).
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block GRID body composes: the
// loop op (fold_model "scalar_delta_grid") + the iq1_m ternary-grid integer core brick
// + the one-operand scalar yield round-trip (the region carries the (index, f32) pair
// and the yield names the single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.iq1_m_q8_k_grid_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The honest body lowers to a REAL emitc.func -- the byte-exact SCALAR-accumulator
// GRID emit, byte-identical to the (now-retired) monolith emitIQ1MQ8KBlockDot (same
// grid decl + per-super-block grid body helper, same facts, same order) modulo the
// source-op provenance token + the func name.
// EMIT: emitc.func @tcrv_emitc_iq1_m_super_block_grid_core_kernel_iq1_m_super_block_grid_core(
// The 2048-entry TERNARY GRID codebook emitted ONCE as a structured static const
// decl (keyed off the grid-core brick op identity from the canonical kIQ1MGrid); the
// iq1_m decl NAME is distinct from iq1_s (tcrv_iq1m_grid vs tcrv_iq1s_grid).
// EMIT: verbatim "static const uint64_t tcrv_iq1m_grid[2048] = {0xffffffffffffffffULL,
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the
// loop (NO 8-lane sums vector).
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the
// grid core brick operands: vx + ib*56, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "56"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The packed iq1m_scale fp16 RECONSTRUCT (the NEW piece vs iq1_s: NO fp16 d field --
// the 4 scales[] words OR'd into a uint16 lvalue read as _Float16).
// EMIT: callee=iq1m_scale_reconstruct
// EMIT: bitwise_or
// EMIT: call_opaque "(float)*(const _Float16 *)"
// The TWO int32 SCALAR accumulators sumi1 + sumi2 (NO aux32 memory state).
// EMIT: !emitc.lvalue<!emitc.opaque<"int32_t">>
// The per-group Σq8 DELTA term (bsums unusable -- FOUR independent signs; grid-
// independent i8m1 vwredsum of the loaded q8 vector).
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// The HALF-SPLIT vluxei16 ternary-grid gather (iq1_m gathers 2 u64 per half -> i64m1,
// reinterpret i8m1 = 16 grid bytes, i16m2 product, ONE vwredsum per half -- DISTINCT
// from iq1_s's whole-sub-block i64m2/i8m2/i16m4 gather).
// EMIT: call_opaque "__riscv_vle16_v_u16mf4"
// EMIT: call_opaque "__riscv_vluxei16_v_i64m1"
// EMIT: call_opaque "__riscv_vreinterpret_v_i64m1_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The per-super-block scalar delta fold `sumf += d*((float)sumi1 + 0.125f*
// (float)sumi2)` as ONE emitc.expression with the 0.125f (IQ1M_DELTA) literal. NO
// 8-lane vfcvt/vfmul, NO post-loop vse32 horizontal sum.
// EMIT: literal "0.125f"
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vse32_v_f32m2"
// The *s store (NO trailing factor -- *s = sumf).
// EMIT: callee=store_s
// iq1_m must NOT emit iq1_s's whole-sub-block gather widths (would mean wrong dispatch).
// EMIT-NOT: call_opaque "__riscv_vluxei16_v_i64m2"
// EMIT-NOT: verbatim "static const uint64_t tcrv_iq1s_grid

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the grid core's q8 base
// swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation
// arithmetic collapses (both bases fold to super_block_base_x off the weight base via
// the per-body memo), so no super_block_base_y appears between super_block_base_x and
// the packed-scale reconstruct. The emit provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=iq1m_scale_reconstruct

// MGATE anti-bypass (the operand-driven driver gate): dropping the grid core's
// block_index parses + verifies clean (the standalone 4-operand form) but FAILS to
// legalize -- the driver requires the addressing brick's block_index to be the loop
// induction variable (region arg 0), so a body that would silently address
// super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'tcrv.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block GRID loop
// region is fail-closed rejected (the recursive [L-8] allowlist default is deny,
// naming the offending op) -- the iq1_m grid core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7).
// The SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed.
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator iq1_s fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// SCALARYIELDSECOND: must NOT carry a second
// BADSTRIDE: requires weight_block_stride == 56 (sizeof block_iq1_m
