// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's|// R2 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT-NESTED
// M-FLAT milestone-2 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not
// gate-only): (MSWAP) swapping BRICK 3's q8 activation base %vy -> %vx CHANGES
// the emit -- the super_block_base_y activation-base arithmetic no longer
// precedes the Region-A unpack (the scaled dot folds off the WEIGHT base now),
// so a valid operand edit yields DIFFERENT C (not a rejection). (MGATE) dropping
// BRICK 1's block_index still PARSES + VERIFIES but FAILS to legalize -- the
// driver gate requires every brick's block_index to be the loop induction
// variable so the emit addresses base + ib*stride, never super-block-0.
// RUN: sed 's/q4_k_scaled_dot %%aux8, %%scales, %%vy,/q4_k_scaled_dot %%aux8, %%scales, %%vx,/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/q4_k_nibble_unpack %%vx, %%vl block %%super_block_index : index/q4_k_nibble_unpack %%vx, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "super_block_two_level_scale_min"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's|, fold_model = "super_block_two_level_scale_min"||' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=MISSINGFOLD
// RUN: sed 's/scale_min", integer_core_lmul = "m2"/scale_min", integer_core_lmul = "m8"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADLMUL
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS
// RUN: sed 's/loop_yield %%sums, %%sumf : !weft_rvv.vector<f32, "m2">, f32/loop_yield %%sumf, %%sums : f32, !weft_rvv.vector<f32, "m2">/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=SWAPYIELD

// M-FLAT q4_K super-block scaffold (milestone-1: W1+W3+W2+W4) -- the FIRST typed
// loop op that carries a DUAL accumulator. weft_rvv.typed_super_block_block_dot_
// loop_body records the q4_K/q5_K super-block dot-product's OUTER nb = n/QK_K loop
// as ONE region-carrying op whose entry arguments are the super_block_index
// induction variable AND two loop-carried accumulators the flat single-f32 loop
// op cannot express: the `sums` 8-lane fp32 VECTOR accumulator (the deferred
// positive `sums += d*(float)aux32` fold, BRICK 6) and the `sumf` scalar fp32
// accumulator (the in-loop `sumf -= dmin*Σ(mins*bsums)` MIN term, BRICK 4). The
// region is terminated by weft_rvv.typed_super_block_block_dot_loop_yield naming
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
  weft.exec.kernel @q4_k_super_block_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @q4_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // Placeholder scratch runtime_abi_values standing in for the per-super-block
      // aux8/scales/aux32 scratch the bricks read/write (the real scratch wiring
      // is a later step; the scaffold only needs the brick operand C types to
      // verify).
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %aux32 = weft_rvv.runtime_abi_value {c_name = "aux32", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q4-aux32-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q4_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, fold_model = "super_block_two_level_scale_min", integer_core_lmul = "m2"} {
        ^bb0(%super_block_index: index, %sums: !weft_rvv.vector<f32, "m2">, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // R2 %r2vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
          // R2 weft_rvv.with_vl %r2vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
          // R2   %r2 = weft_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %r2vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "mf4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // R2 } : !weft_rvv.vl
          // BRICK 1: plain 4-bit nibble unpack -> aux8[256] scratch (Region A).
          // The `block %super_block_index` operand (W5) makes the weight base
          // per-super-block (vx + ib*144), not loop-invariant super-block-0.
          %b1 = weft_rvv.q4_k_nibble_unpack %vx, %vl block %super_block_index : index {kind = "q4_k_nibble_unpack", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_qs_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 2: 6-bit scale/min bit-dance -> utmp[4] scratch (Region B).
          %b2 = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl block %super_block_index : index {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 3: per-sub-block uint6-scaled i32 dot + fold-back (Region C).
          // activation_quant_byte_offset = 4 places the q8 strip at yb + 4 (after
          // the fp32 activation d), the loop-form q8 offset (W5).
          %b3 = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl block %super_block_index : index {kind = "q4_k_scaled_dot", integer_core_lmul = "m2", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 4: MIN term (sumf -= dmin * Σ(mins * bsums)) -- the SCALAR sumf
          // accumulator chain.
          %b4 = weft_rvv.q4_k_min_term %vx, %scales, %vy, %vl block %super_block_index : index {kind = "q4_k_min_term", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, bsums_byte_offset = 260 : i64, weight_dmin_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 6: deferred positive fold (sums += d * (float)aux32) -- the
          // 8-lane fp32 VECTOR sums accumulator chain.
          %b6 = weft_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl block %super_block_index : index {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // The DUAL carried-out accumulators (sums vector + sumf scalar). The
          // byte-exact fold expressions that update them are a later step; the
          // scaffold carries them through the region to the dual yield.
          weft_rvv.typed_super_block_block_dot_loop_yield %sums, %sumf : !weft_rvv.vector<f32, "m2">, f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse and verify -- the DUAL-accumulator super-block loop op + the two-operand
// yield round-trip (the region carries the (index, vector<f32 m2>, f32) triple
// and the yield names both carried-out accumulators).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: !weft_rvv.vector<f32, "m2">, %{{.*}}: f32):
// VERIFY: weft_rvv.q4_k_nibble_unpack
// VERIFY: weft_rvv.q4_k_scale_min_bit_dance
// VERIFY: weft_rvv.q4_k_scaled_dot
// VERIFY: weft_rvv.q4_k_min_term
// VERIFY: weft_rvv.q4_k_sums_fold_scale_d
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}} : !weft_rvv.vector<f32, "m2">, f32

// Milestone-2 (W5-W7): the honest body now lowers to a REAL emitc.func through
// the weft.exec.variant legalization -- the byte-exact DUAL-accumulator emit. It
// clears the recursive [L-8] allowlist gate (never emitting the allowlist
// rejection) AND materializes the full super-block block dot. The emit is
// byte-identical to the monolithic emitQ4_KQ8_KBlockDot (same shared CORE
// helpers, same facts, same order) modulo the source-op provenance token; the
// structural landmarks below assert the W5-W7 shape.
// EMIT: emitc.func @weft_emitc_q4_k_super_block_loop_body_kernel_q4_k_super_block_loop_body(
// W6 DUAL accumulator: the `sums` vfloat32m2 + `sumf` float emitc.variable
// declared + seeded ONCE outside the super-block loop.
// EMIT: local_variable=sums
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: local_variable=sumf
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// W5 per-super-block addressing (x weight-base then y activation-base, built
// from the brick operands: vx + ib*144, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "144"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// W7 Region A/B/C integer core (shared leaf helpers, m2 widening chain):
// EMIT: callee=unpack_4bit
// EMIT: call_opaque "__riscv_vmv_v_x_i32m8"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m4"
// EMIT: call_opaque "__riscv_vwmacc_vx_i32m8"
// W7 MIN term + deferred positive fold (the SEPARATE vfmul/vfadd, no fma):
// EMIT: callee=fold_activation_d
// EMIT: callee=min_term_bsums
// EMIT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT: call_opaque "__riscv_vfmul_vf_f32m2"
// EMIT: call_opaque "__riscv_vfadd_vv_f32m2"
// W6 post-loop vector->scalar horizontal fold (BRICK 7 helper) + *s store:
// EMIT: callee=store_sums_lanes
// EMIT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT: callee=horizontal_sum
// EMIT: callee=store_s

// MSWAP anti-bypass (operand-driven, NOT gate-only): with BRICK 3's q8 base
// swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation
// arithmetic no longer precedes the Region-A unpack (the scaled dot folds off
// the weight base), so no super_block_base_y appears between super_block_base_x
// and the first unpack_4bit. This would be FALSE for the honestly-wired body
// (which emits super_block_base_y right there), so the emit provably tracks the
// q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=unpack_4bit

// MGATE anti-bypass (the operand-driven driver gate): dropping BRICK 1's
// block_index parses + verifies clean but FAILS to legalize -- the driver
// requires every addressing brick's block_index to be the loop induction
// variable (region arg 0), so a body that would silently address super-block-0
// is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the super-block loop region is fail-closed
// rejected (the allowlist default is deny, naming the offending op).
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// A hand-written monolithic *_block_dot helper NESTED inside an allowlisted
// with_vl inside the super-block loop region is fail-closed rejected ONLY because
// the allowlist walk is RECURSIVE -- the literal [L-8] opaque-helper leak the
// strong-form gate exists to catch.
// REJECT-NESTED: 'weft_rvv.q4_0_q8_0_block_dot' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind, the W2 fold_model fact, and
// the integer_core_lmul scheduling knob (I7). The DUAL-accumulator region/yield
// contract is fail-closed on arg count and on a swapped (scalar-first) yield.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// MISSINGFOLD: requires attribute 'fold_model'
// BADLMUL: only accepts integer_core_lmul "mf2", "m1", or "m2"
// BADARGS: requires the region to carry exactly three entry arguments
// SWAPYIELD: requires the loop yield to carry an 8-lane fp32 vector `sums` accumulator
