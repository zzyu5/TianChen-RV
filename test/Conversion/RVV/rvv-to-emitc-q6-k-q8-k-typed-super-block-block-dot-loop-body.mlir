// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// M-FLAT q6_K milestone-2 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not
// gate-only): (MSWAP) swapping the aux32 core's q8 activation base %vy -> %vx
// CHANGES the emit -- the super_block_base_y activation-base arithmetic no longer
// precedes the Region-A unpack (the core folds off the WEIGHT base now), so a
// valid operand edit yields DIFFERENT C (not a rejection). (MGATE) dropping the
// aux32 core's block_index still PARSES + VERIFIES but FAILS to legalize -- the
// driver gate requires every brick's block_index to be the loop induction
// variable so the emit addresses base + ib*stride, never super-block-0.
// RUN: sed 's/q6_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out/q6_k_q8_k_aux32_partial %%vx, %%vx, %%aux32out/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/q6_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out, %%n, %%vl block %%super_block_index : index/q6_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out, %%n, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sums: !weft_rvv.vector<f32, "m2">):/%%sums: !weft_rvv.vector<f32, "m2">, %%sumf: f32):/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SINGLE
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "super_block_two_level_scale_min"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sums : !weft_rvv.vector<f32, "m2">/loop_yield %%sums, %%spare_sumf : !weft_rvv.vector<f32, "m2">, f32/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=SINGLEYIELDSUMF

// M-FLAT q6_K super-block SINGLE-accumulator emit (milestone-2: W-D byte-exact
// single-accumulator lowering + the reused q6_K aux32 integer core). Unlike the
// q4_K/q5_K DUAL-accumulator super-block loop (sums vector + sumf scalar MIN
// chain), q6_K has NO per-block min => the fold folds ONLY the `sums` 8-lane fp32
// VECTOR (post-loop horizontal add), with no scalar MIN chain.
// weft_rvv.typed_super_block_block_dot_loop_body with fold_model =
// "scales_times_sumi" therefore carries a SINGLE accumulator: the region entry
// arguments are just (super_block_index, sums) and the region is terminated by
// weft_rvv.typed_super_block_block_dot_loop_yield naming the `sums` vector ALONE
// (no sumf operand). The honest target body carries the q6_K INTEGER CORE
// (weft_rvv.q6_k_q8_k_aux32_partial -- the 2-bit qh + 8-bit signed scale unpack
// into aux32) with a per-super-block `block %super_block_index` operand, followed
// by the REUSED no-min positive fold (weft_rvv.q4_k_sums_fold_scale_d: sums +=
// fp16(x.d @208)*y.d*(float)aux32, also per-super-block), then the single yield.
//
// This is the SOLE q6_K super-block lowering (the opaque monolith emitQ6_KQ8_KBlockDot
// is retired the same action as this flip). The emit is byte-identical to that
// retired monolith (same q6_K aux32 core helper, same facts, same order) modulo
// the source-op provenance token; the structural landmarks below assert the W-D
// shape. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested
// here. This is additive: it never touches the q4_K/q5_K DUAL-accumulator path
// (their fold_model branch is unchanged -- zero regression).

module {
  weft.exec.kernel @q6_k_super_block_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @q6_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // Placeholder scratch runtime_abi_values standing in for the per-super-block
      // aux32 integer-state scratch the bricks write/read (the real scratch is a
      // function-scoped variable the loop emitter declares itself; in the loop form
      // these operand slots are vestigial, but the standalone-form C types are pinned
      // so the MGATE anti-bypass -- dropping block_index -> standalone form -- still
      // PARSES + VERIFIES before the driver gate fails legalization). %aux32out is
      // the aux32 core's int32_t* output; %aux32 is the fold brick's const int32_t*
      // aux32 slot.
      %aux32out = weft_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "q6-aux32-scratch-out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %aux32 = weft_rvv.runtime_abi_value {c_name = "aux32r", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q6-aux32-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q6_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q6_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, fold_model = "scales_times_sumi"} {
        ^bb0(%super_block_index: index, %sums: !weft_rvv.vector<f32, "m2">):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // q6_K INTEGER CORE: the 2-bit qh + 8-bit signed scale unpack into the
          // per-super-block aux32[8] i32 state. The `block %super_block_index`
          // operand makes the weight/activation bases per-super-block (vx + ib*210,
          // vy + ib*292), not loop-invariant super-block-0.
          %q6core = weft_rvv.q6_k_q8_k_aux32_partial %vx, %vy, %aux32out, %n, %vl block %super_block_index : index {kind = "ggml_q6_k_q8_k_aux32_partial", scale_model = "per-sub-block-int8-scale-i32-domain", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, weight_qh_byte_offset = 128 : i64, weight_scales_byte_offset = 192 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // REUSED no-min positive fold: sums += fp16(x.d @208)*y.d*(float)aux32 --
          // the 8-lane fp32 VECTOR sums accumulator chain, NO min term (q6_K has no
          // per-block min). weight_d_byte_offset = 208 = block_q6_K d offset.
          // Per-super-block via block_index.
          %b6 = weft_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl block %super_block_index : index {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 208 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // SINGLE carried-out accumulator (sums vector ONLY -- no sumf).
          weft_rvv.typed_super_block_block_dot_loop_yield %sums : !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse and verify -- the SINGLE-accumulator super-block loop op + the one-operand
// yield round-trip (the region carries the (index, vector<f32 m2>) pair and the
// yield names the single carried-out `sums` vector, NO sumf).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: !weft_rvv.vector<f32, "m2">):
// VERIFY: weft_rvv.q6_k_q8_k_aux32_partial %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.q4_k_sums_fold_scale_d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : !weft_rvv.vector<f32, "m2">
// VERIFY-NOT: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// Milestone-2 (W-D): the honest body lowers to a REAL emitc.func -- the byte-exact
// SINGLE-accumulator emit, byte-identical to the retired monolith emitQ6_KQ8_KBlockDot
// (same q6_K aux32 core helper, same facts, same order) modulo the source-op
// provenance token.
// EMIT: emitc.func @weft_emitc_q6_k_super_block_loop_body_kernel_q6_k_super_block_loop_body(
// The SINGLE `sums` vfloat32m2 emitc.variable, seeded ONCE outside the loop.
// EMIT: local_variable=sums
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the
// aux32 core brick operands: vx + ib*210, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "210"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The q6_K integer core: the 2-bit qh + 8-bit signed scale unpack (unpack_6bit,
// the `vand_vx`/`vsll_vx` qh dance) then the per-sub-block int8-scaled vwmacc into
// the 8-lane aux32 (mf2 chain: i16m1 product, i32m2 accumulator).
// EMIT: callee=unpack_6bit
// EMIT: call_opaque "__riscv_vmv_v_x_i32m2"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m1"
// EMIT: call_opaque "__riscv_vwmacc_vx_i32m2"
// The no-min positive fold (the SEPARATE vfmul/vfadd, no fma), reading the q6_K
// weight d at byte offset 208.
// EMIT: callee=fold_scale_d
// EMIT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT: call_opaque "__riscv_vfmul_vf_f32m2"
// EMIT: call_opaque "__riscv_vfadd_vv_f32m2"
// NO MIN chain (the single-accumulator no-min path): no min-term bsums dot, no
// dmin subtraction, no fused fp fma.
// EMIT-NOT: callee=min_term
// EMIT-NOT: callee=min_subtract
// EMIT-NOT: callee=fold_scale_dmin
// EMIT-NOT: call_opaque "__riscv_vfmacc
// EMIT-NOT: call_opaque "__riscv_vfmadd
// Post-loop SEQUENTIAL horizontal add (NOT a vfredusum) + the *s store.
// EMIT: callee=store_sums_lanes
// EMIT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// EMIT: callee=horizontal_sum
// EMIT: callee=store_s

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the aux32 core's q8 base
// swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation
// arithmetic no longer precedes the Region-A unpack (the core folds off the weight
// base), so no super_block_base_y appears between super_block_base_x and the first
// unpack_6bit. This would be FALSE for the honestly-wired body (which emits
// super_block_base_y right there), so the emit provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=unpack_6bit

// MGATE anti-bypass (the operand-driven driver gate): dropping the aux32 core's
// block_index parses + verifies clean (the standalone-form int32_t* output type is
// honored) but FAILS to legalize -- the driver requires every addressing brick's
// block_index to be the loop induction variable (region arg 0), so a body that
// would silently address super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the single-accumulator super-block loop region
// is fail-closed rejected (the recursive [L-8] allowlist default is deny, naming
// the offending op) -- the q6_K aux32 core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7).
// The SINGLE-accumulator region/yield contract is fail-closed and fold_model-keyed:
// the no-min "scales_times_sumi" path rejects a dual (index, sums, sumf) region and
// a dual (sumf-carrying) yield; the dual "super_block_two_level_scale_min" path
// rejects this no-min (index, sums) region.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SINGLE: requires the region to carry exactly two entry arguments for the single-accumulator no-min fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// SINGLEYIELDSUMF: must NOT carry a `sumf` scalar accumulator in the yield
