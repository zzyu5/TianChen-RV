// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=PASTGATE
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's|// R2 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT-NESTED
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "super_block_two_level_scale_min"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's|, fold_model = "super_block_two_level_scale_min"||' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=MISSINGFOLD
// RUN: sed 's/scale_min", integer_core_lmul = "m2"/scale_min", integer_core_lmul = "m8"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADLMUL
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS
// RUN: sed 's/loop_yield %%sums, %%sumf : !tcrv_rvv.vector<f32, "m2">, f32/loop_yield %%sumf, %%sums : f32, !tcrv_rvv.vector<f32, "m2">/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SWAPYIELD

// M-FLAT q4_K super-block scaffold (milestone-1: W1+W3+W2+W4) -- the FIRST typed
// loop op that carries a DUAL accumulator. tcrv_rvv.typed_super_block_block_dot_
// loop_body records the q4_K/q5_K super-block dot-product's OUTER nb = n/QK_K loop
// as ONE region-carrying op whose entry arguments are the super_block_index
// induction variable AND two loop-carried accumulators the flat single-f32 loop
// op cannot express: the `sums` 8-lane fp32 VECTOR accumulator (the deferred
// positive `sums += d*(float)aux32` fold, BRICK 6) and the `sumf` scalar fp32
// accumulator (the in-loop `sumf -= dmin*Σ(mins*bsums)` MIN term, BRICK 4). The
// region is terminated by tcrv_rvv.typed_super_block_block_dot_loop_yield naming
// BOTH carried-out accumulators. The honest target body carries the 5 in-loop
// q4_K bricks (nibble_unpack -> scale_min_bit_dance -> scaled_dot -> min_term ->
// sums_fold_scale_d) over placeholder scratch runtime_abi_values, plus the dual
// accumulator and the two-level fold_model.
//
// This milestone proves the SCAFFOLD only: PARSE + VERIFY (the dual-accumulator
// region/yield verifier, the W2 super_block_two_level_scale_min fold_model, the
// bounded-surface fail-closed gates) and that the honest body clears the
// recursive [L-8] allowlist gate (W4: the 6 q4_K bricks are allowlist members).
// The byte-exact dual-accumulator lowering is a LATER step (W5-W7); the pass
// therefore stops at "failed to legalize the variant" AFTER clearing the gate,
// never emitting the allowlist rejection. Numerical bit-exact-vs-ggml is
// pending-hardware (ssh rvv), not tested here. This is additive: it never touches
// the flat loop op / single-block strong paths (zero regression).

module {
  tcrv.exec.kernel @q4_k_super_block_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // Placeholder scratch runtime_abi_values standing in for the per-super-block
      // aux8/scales/aux32 scratch the bricks read/write (the real scratch wiring
      // is a later step; the scaffold only needs the brick operand C types to
      // verify).
      %aux8 = tcrv_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked-scratch", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scales = tcrv_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales-scratch", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %aux32 = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q4-aux32-scratch", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q4_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, fold_model = "super_block_two_level_scale_min", integer_core_lmul = "m2"} {
        ^bb0(%super_block_index: index, %sums: !tcrv_rvv.vector<f32, "m2">, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // R2 %r2vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
          // R2 tcrv_rvv.with_vl %r2vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
          // R2   %r2 = tcrv_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %r2vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "mf4"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // R2 } : !tcrv_rvv.vl
          // BRICK 1: plain 4-bit nibble unpack -> aux8[256] scratch (Region A).
          %b1 = tcrv_rvv.q4_k_nibble_unpack %vx, %vl {kind = "q4_k_nibble_unpack", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_qs_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // BRICK 2: 6-bit scale/min bit-dance -> utmp[4] scratch (Region B).
          %b2 = tcrv_rvv.q4_k_scale_min_bit_dance %vx, %vl {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // BRICK 3: per-sub-block uint6-scaled i32 dot + fold-back (Region C).
          %b3 = tcrv_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "m2", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // BRICK 4: MIN term (sumf -= dmin * Σ(mins * bsums)) -- the SCALAR sumf
          // accumulator chain.
          %b4 = tcrv_rvv.q4_k_min_term %vx, %scales, %vy, %vl {kind = "q4_k_min_term", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, bsums_byte_offset = 260 : i64, weight_dmin_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // BRICK 6: deferred positive fold (sums += d * (float)aux32) -- the
          // 8-lane fp32 VECTOR sums accumulator chain.
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // The DUAL carried-out accumulators (sums vector + sumf scalar). The
          // byte-exact fold expressions that update them are a later step; the
          // scaffold carries them through the region to the dual yield.
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums, %sumf : !tcrv_rvv.vector<f32, "m2">, f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse and verify -- the DUAL-accumulator super-block loop op + the two-operand
// yield round-trip (the region carries the (index, vector<f32 m2>, f32) triple
// and the yield names both carried-out accumulators).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: !tcrv_rvv.vector<f32, "m2">, %{{.*}}: f32):
// VERIFY: tcrv_rvv.q4_k_nibble_unpack
// VERIFY: tcrv_rvv.q4_k_scale_min_bit_dance
// VERIFY: tcrv_rvv.q4_k_scaled_dot
// VERIFY: tcrv_rvv.q4_k_min_term
// VERIFY: tcrv_rvv.q4_k_sums_fold_scale_d
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}} : !tcrv_rvv.vector<f32, "m2">, f32

// The honest body CLEARS the recursive [L-8] allowlist gate (the 6 q4_K bricks +
// the super-block loop op/yield are allowlist members): the pass proceeds PAST
// the gate to the conversion stage and stops ONLY because the byte-exact
// dual-accumulator lowering is a later step (W5-W7) -- it never emits the
// allowlist rejection.
// PASTGATE: failed to legalize operation 'tcrv.exec.variant'
// PASTGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the super-block loop region is fail-closed
// rejected (the allowlist default is deny, naming the offending op).
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// A hand-written monolithic *_block_dot helper NESTED inside an allowlisted
// with_vl inside the super-block loop region is fail-closed rejected ONLY because
// the allowlist walk is RECURSIVE -- the literal [L-8] opaque-helper leak the
// strong-form gate exists to catch.
// REJECT-NESTED: 'tcrv_rvv.q4_0_q8_0_block_dot' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind, the W2 fold_model fact, and
// the integer_core_lmul scheduling knob (I7). The DUAL-accumulator region/yield
// contract is fail-closed on arg count and on a swapped (scalar-first) yield.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// MISSINGFOLD: requires attribute 'fold_model'
// BADLMUL: only accepts integer_core_lmul "mf2", "m1", or "m2"
// BADARGS: requires the region to carry exactly three entry arguments
// SWAPYIELD: requires the loop yield to carry an 8-lane fp32 vector `sums` accumulator
