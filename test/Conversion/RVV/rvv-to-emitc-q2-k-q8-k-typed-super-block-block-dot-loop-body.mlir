// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=GATE
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "scalar_scale_min"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_scale_min"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// M-FLAT q2_K super-block scaffold (milestone-1: W-A SCALAR-accumulator loop op
// variant + W-B fold_model + W-C q2_K integer-core brick + W-C' allowlist). q2_K
// is a super-block affine quant (it HAS a min term, like q4_K/q5_K) but its fold
// is STRUCTURALLY DISTINCT from every K-quant translated so far: q2_K accumulates
// a SINGLE per-super-block scalar `sumf += dall*isum - dmin*summs`, where isum and
// summs are SCALAR integer states -- NOT q4_K/q5_K's 8-lane deferred VECTOR `sums`
// + scalar `sumf` DUAL, and NOT q6_K's 8-lane VECTOR `sums`-only SINGLE. So q2_K's
// loop carries a SCALAR accumulator: tcrv_rvv.typed_super_block_block_dot_loop_body
// with fold_model = "scalar_scale_min" has region entry arguments (super_block_index,
// sumf:f32) and is terminated by tcrv_rvv.typed_super_block_block_dot_loop_yield
// naming the `sumf` SCALAR ALONE (no 8-lane `sums` vector operand). fold_model KEYS
// the arity: dual (index, sums, sumf) + dual yield [q4_K/q5_K]; single-vector
// (index, sums) + single vector yield [q6_K]; scalar (index, sumf) + single scalar
// yield [q2_K, this file].
//
// The honest target body carries the NET-NEW q2_K INTEGER CORE
// (tcrv_rvv.q2_k_q8_k_integer_core -- the 2-bit weight unpack + PLAIN uint4-nibble
// scale/min extraction, NO 6-bit bit-dance, + per-sub-block scalar i32 dot) with a
// per-super-block `block %super_block_index` operand, producing the two SCALAR
// integer states isum + summs. The byte-exact SCALAR fold that updates `sumf`
// (sumf += dall*isum - dmin*summs, dall = fp16(x.d @80)*y.d, dmin = fp16(x.dmin
// @82)*y.d) is a LATER step (M2); the scaffold carries `sumf` through the region to
// the single scalar yield.
//
// This milestone proves the SCAFFOLD only: PARSE + VERIFY (the scalar-accumulator
// region/yield verifier keyed on the W-B "scalar_scale_min" fold_model, the bounded
// -surface fail-closed gates) and that the honest body clears the recursive [L-8]
// allowlist gate (W-C': the q2_K integer core is an allowlist member). The
// byte-exact scalar-accumulator lowering is a LATER step (M2); the pass therefore
// stops at "failed to legalize the variant" AFTER clearing the gate, never emitting
// the allowlist rejection. Numerical bit-exact-vs-ggml is pending-hardware (ssh
// rvv), not tested here. This is additive: it never touches the q4_K/q5_K DUAL nor
// the q6_K SINGLE-VECTOR path (their fold_model branches are unchanged -- zero
// regression).

module {
  tcrv.exec.kernel @q2_k_super_block_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q2_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q2_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q2_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_scale_min", integer_core_lmul = "m1"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // q2_K INTEGER CORE (NET-NEW): the 2-bit weight unpack `(qs >> shift) & 3`
          // + PLAIN uint4-nibble scale/min (sc & 0xF is the scale, sc >> 4 is the
          // min -- NO 6-bit utmp/kmask bit-dance) + per-sub-block scalar i32 dot,
          // producing the two SCALAR integer states isum (Σ (sc&0xF)*subdot) and
          // summs (Σ bsums*(sc>>4)). The `block %super_block_index` operand (W-C)
          // makes the weight/activation bases per-super-block (vx + ib*84,
          // vy + ib*292), not loop-invariant super-block-0.
          %isum, %summs = tcrv_rvv.q2_k_q8_k_integer_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_q2_k_q8_k_integer_core", scale_model = "per-sub-block-uint4-scale-i32-domain-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += dall*isum - dmin*summs
          // (dall = fp16(x.d @80)*y.d, dmin = fp16(x.dmin @82)*y.d) is milestone-2;
          // the scaffold carries `sumf` through the region to the single scalar
          // yield.
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse and verify -- the SCALAR-accumulator super-block loop op + the one-operand
// yield round-trip (the region carries the (index, f32) pair and the yield names
// the single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.q2_k_q8_k_integer_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// [L-8] gate cleared, emit deferred to M2: the honest scalar-accumulator body
// clears the recursive allowlist gate (the q2_K integer core is an allowlist
// member), so the pass NEVER emits an allowlist rejection -- it stops at "failed to
// legalize the variant" because the byte-exact scalar-accumulator lowering is M2
// (not built here; the scalar_scale_min body has no q4_K bricks, so the emitter
// match-fails, leaving the variant illegal).
// GATE: failed to legalize operation 'tcrv.exec.variant'
// GATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block loop region is
// fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the
// offending op) -- the q2_K integer core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the W-B fold_model fact
// (I7). The SCALAR-accumulator region/yield contract is fail-closed and
// fold_model-keyed: the scalar "scalar_scale_min" path rejects an extra region arg
// (a vector-carrying dual/triple region) and a dual (second-operand) yield; the dual
// "super_block_two_level_scale_min" path rejects this scalar (index, sumf) region.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator q2_K fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// SCALARYIELDSECOND: must NOT carry a second
