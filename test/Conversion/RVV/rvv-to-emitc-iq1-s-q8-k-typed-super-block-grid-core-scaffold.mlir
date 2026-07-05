// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=CLEARS
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// M-FLAT iq1_s milestone-1 (the FIRST super-block GRID/codebook typed body):
// composability scaffold for the ggml IQ1_S x Q8_K super-block TERNARY-grid dot.
// iq1_s is a super-block quant whose sub-block decode is a codebook GRID GATHER
// (decode_model=lookup, the 2048-entry ternary iq1s_grid + vluxei16 gather), the
// super-block-flavor sibling of the FLAT codebook iq4_nl. Its fold is a SINGLE
// per-super-block scalar `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)`,
// where sumi (the qh-scaled ternary-grid positive dot) and sumi1 (the delta-bsum
// integer sum) are SCALAR integer states -- the SAME scalar-accumulator arity as
// q2_K's "scalar_scale_min", but a DISTINCT fold arithmetic and a GRID (not
// arithmetic) integer core. So tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model = "scalar_delta_grid" carries a SCALAR accumulator: the region entry
// arguments are just (super_block_index, sumf:f32) and the region is terminated by
// tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE.
// The body carries the NET-NEW iq1_s TERNARY-grid INTEGER CORE
// (tcrv_rvv.iq1_s_q8_k_grid_core -- the 11-bit grid index build from qs+qh, the
// vluxei16 ternary-grid gather, the signed widening grid dot, the qh-encoded
// per-sub-block scale + delta sign, the delta-bsum sum) with a per-super-block
// `block %super_block_index` operand, producing the two SCALAR integer states
// sumi + sumi1.
//
// milestone-1 builds ONLY the ODS + verifier scaffold: the honest target IR
// PARSES + VERIFIES and CLEARS the recursive [L-8] super-block allowlist (the grid
// core is an allowlisted typed pattern-library primitive). The byte-exact
// grid-gather emit (the sub-block vluxei16 decode + the scalar delta fold
// `sumf += d*(sumi + IQ1S_DELTA*sumi1)`, byte-identical to the retired-later
// monolith tcrv_rvv.iq1_s_q8_k_block_dot) is milestone-2; there is NO emit for
// fold_model "scalar_delta_grid" yet, so the lower pass CLEARS the allowlist and
// fails ONLY at legalization. This is additive: it never touches the q4_K/q5_K
// DUAL, the q6_K SINGLE-vector, nor the q2_K scalar path (their fold_model
// branches are unchanged -- zero regression).

module {
  tcrv.exec.kernel @iq1_s_super_block_grid_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @iq1_s_super_block_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq1_s_super_block_grid_core, sew = 32 : i64, source_kernel = "iq1_s_super_block_grid_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
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
          %sumi, %sumi1 = tcrv_rvv.iq1_s_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq1_s_q8_k_grid_core", scale_model = "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 2 : i64, weight_qh_byte_offset = 34 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += d*((float)sumi +
          // IQ1S_DELTA*(float)sumi1) (d = fp16(x.d @0)*y.d @0, IQ1S_DELTA=0.125f)
          // is milestone-2 (emitter-inlined, no separate fold brick).
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block GRID body composes: the
// loop op (fold_model "scalar_delta_grid") + the ternary-grid integer core brick +
// the one-operand scalar yield round-trip (the region carries the (index, f32)
// pair and the yield names the single carried-out `sumf` scalar, NO 8-lane `sums`
// vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.iq1_s_q8_k_grid_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// milestone-1 CLEARS the recursive [L-8] super-block allowlist (the grid core is a
// registered typed pattern-library primitive) and fails ONLY at legalization --
// there is NO emit for fold_model "scalar_delta_grid" yet (the byte-exact
// grid-gather emit is milestone-2). The failure is a legalization failure, NOT an
// allowlist rejection.
// CLEARS: failed to legalize operation 'tcrv.exec.variant'
// CLEARS-NOT: allowlist

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
