// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=GATE
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sums: !tcrv_rvv.vector<f32, "m2">):/%%sums: !tcrv_rvv.vector<f32, "m2">, %%sumf: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SINGLE
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sums : !tcrv_rvv.vector<f32, "m2">/loop_yield %%sums, %%spare_sumf : !tcrv_rvv.vector<f32, "m2">, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SINGLEYIELDSUMF

// M-FLAT q6_K super-block scaffold (milestone-1: W-A single-accumulator loop op
// variant + W-B fold_model + W-C allowlist). Unlike the q4_K/q5_K DUAL-accumulator
// super-block loop (sums vector + sumf scalar MIN chain), q6_K has NO per-block
// min => the fold folds ONLY the `sums` 8-lane fp32 VECTOR (post-loop horizontal
// add), with no scalar MIN chain. tcrv_rvv.typed_super_block_block_dot_loop_body
// with fold_model = "scales_times_sumi" therefore carries a SINGLE accumulator:
// the region entry arguments are just (super_block_index, sums) and the region is
// terminated by tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sums`
// vector ALONE (no sumf operand). fold_model KEYS the arity: the two-level path is
// (index, sums, sumf) + dual yield, this no-min path is (index, sums) + single
// yield.
//
// The honest target body carries the q6_K INTEGER CORE (tcrv_rvv.q6_k_q8_k_aux32_
// partial -- the 2-bit qh + 8-bit signed scale unpack into aux32) with a per-super-
// block `block %super_block_index` operand, followed by the REUSED q4_K BRICK 6
// no-min positive fold (tcrv_rvv.q4_k_sums_fold_scale_d: sums += d*(float)aux32,
// also per-super-block) over placeholder scratch runtime_abi_values, then the
// single yield.
//
// This milestone proves the SCAFFOLD only: PARSE + VERIFY (the single-accumulator
// region/yield verifier keyed on the W-B "scales_times_sumi" fold_model, the
// bounded-surface fail-closed gates) and that the honest body clears the recursive
// [L-8] allowlist gate (W-C: the q6_K aux32 integer core + the reused q4_K
// sums-fold brick are allowlist members). The byte-exact single-accumulator
// lowering is a LATER step (M2); the pass therefore stops at "failed to legalize
// the variant" AFTER clearing the gate, never emitting the allowlist rejection.
// Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. This
// is additive: it never touches the q4_K/q5_K DUAL-accumulator path (their
// fold_model branch is unchanged -- zero regression).

module {
  tcrv.exec.kernel @q6_k_super_block_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q6_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // Placeholder scratch runtime_abi_values standing in for the per-super-block
      // aux32 integer-state scratch the bricks write/read (the real scratch wiring
      // is a later step; the scaffold only needs the brick operand C types to
      // verify). %aux32out is the aux32 core's int32_t* output; %aux32 is the fold
      // brick's (loop-form-vestigial) const int32_t* aux32 slot.
      %aux32out = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "q6-aux32-scratch-out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %aux32 = tcrv_rvv.runtime_abi_value {c_name = "aux32r", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q6-aux32-scratch", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q6_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q6_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, fold_model = "scales_times_sumi", integer_core_lmul = "m2"} {
        ^bb0(%super_block_index: index, %sums: !tcrv_rvv.vector<f32, "m2">):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // q6_K INTEGER CORE: the 2-bit qh + 8-bit signed scale unpack into the
          // per-super-block aux32[8] i32 state. The `block %super_block_index`
          // operand (W-C) makes the weight/activation bases per-super-block
          // (vx + ib*210, vy + ib*292), not loop-invariant super-block-0.
          %q6core = tcrv_rvv.q6_k_q8_k_aux32_partial %vx, %vy, %aux32out, %n, %vl block %super_block_index : index {kind = "ggml_q6_k_q8_k_aux32_partial", scale_model = "per-sub-block-int8-scale-i32-domain", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, weight_qh_byte_offset = 128 : i64, weight_scales_byte_offset = 192 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // REUSED q4_K BRICK 6 no-min positive fold: sums += fp16(x.d)*y.d*
          // (float)aux32 -- the 8-lane fp32 VECTOR sums accumulator chain, NO min
          // term (q6_K has no per-block min). Per-super-block via block_index.
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl block %super_block_index : index {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // SINGLE carried-out accumulator (sums vector ONLY -- no sumf). The
          // byte-exact fold expression that updates it is a later step; the
          // scaffold carries it through the region to the single yield.
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse and verify -- the SINGLE-accumulator super-block loop op + the one-operand
// yield round-trip (the region carries the (index, vector<f32 m2>) pair and the
// yield names the single carried-out `sums` vector, NO sumf).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: !tcrv_rvv.vector<f32, "m2">):
// VERIFY: tcrv_rvv.q6_k_q8_k_aux32_partial %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.q4_k_sums_fold_scale_d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : !tcrv_rvv.vector<f32, "m2">
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// [L-8] gate cleared, emit deferred to M2: the honest single-accumulator body
// clears the recursive allowlist gate (the q6_K aux32 integer core + the reused
// q4_K sums-fold brick are allowlist members), so the pass NEVER emits an
// allowlist rejection -- it stops at "failed to legalize the variant" because the
// byte-exact single-accumulator lowering is M2 (not built here).
// GATE: failed to legalize operation 'tcrv.exec.variant'
// GATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the single-accumulator super-block loop region
// is fail-closed rejected (the recursive [L-8] allowlist default is deny, naming
// the offending op) -- the q6_K aux32 core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the W-B fold_model fact
// (I7). The SINGLE-accumulator region/yield contract is fail-closed and
// fold_model-keyed: the no-min "scales_times_sumi" path rejects a dual (index,
// sums, sumf) region and a dual (sumf-carrying) yield; the dual
// "super_block_two_level_scale_min" path rejects this no-min (index, sums) region.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SINGLE: requires the region to carry exactly two entry arguments for the single-accumulator no-min fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// SINGLEYIELDSUMF: must NOT carry a `sumf` scalar accumulator in the yield
