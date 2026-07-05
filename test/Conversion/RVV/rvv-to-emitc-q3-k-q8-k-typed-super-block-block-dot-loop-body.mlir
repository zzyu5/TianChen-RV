// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// q3_K first-flip anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only):
// (MSWAP) swapping the aux32 core's q8 activation base %vy -> %vx CHANGES the emit
// -- the super_block_base_y activation-base arithmetic no longer precedes the
// Region-A unpack (the core folds off the WEIGHT base now), so a valid operand edit
// yields DIFFERENT C (not a rejection). (MGATE) dropping the aux32 core's
// block_index still PARSES + VERIFIES but FAILS to legalize -- the driver gate
// requires every brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/q3_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out/q3_k_q8_k_aux32_partial %%vx, %%vx, %%aux32out/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/q3_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out, %%n, %%vl block %%super_block_index : index/q3_k_q8_k_aux32_partial %%vx, %%vy, %%aux32out, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sums: !tcrv_rvv.vector<f32, "m2">):/%%sums: !tcrv_rvv.vector<f32, "m2">, %%sumf: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SINGLE
// RUN: sed 's/fold_model = "scales_times_sumi"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sums : !tcrv_rvv.vector<f32, "m2">/loop_yield %%sums, %%spare_sumf : !tcrv_rvv.vector<f32, "m2">, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SINGLEYIELDSUMF

// q3_K super-block SINGLE-accumulator emit (first flip: the byte-exact
// single-accumulator lowering REUSING q6_K's no-min single-vector arity + fold,
// with the net-new q3_K aux32 integer core). q3_K is SYMMETRIC (NO per-block min),
// like q6_K => the fold folds ONLY the `sums` 8-lane fp32 VECTOR (post-loop
// horizontal add), with no scalar MIN chain. tcrv_rvv.typed_super_block_block_dot
// _loop_body with fold_model = "scales_times_sumi" therefore carries a SINGLE
// accumulator: the region entry arguments are just (super_block_index, sums) and
// the region is terminated by tcrv_rvv.typed_super_block_block_dot_loop_yield naming
// the `sums` vector ALONE (no sumf operand). The honest target body carries the
// q3_K INTEGER CORE (tcrv_rvv.q3_k_q8_k_aux32_partial -- the 2-bit + SUBTRACTIVE-
// hmask decode + the SIGNED 6-bit scale dance into aux32) with a per-super-block
// `block %super_block_index` operand, followed by the REUSED no-min positive fold
// (tcrv_rvv.q4_k_sums_fold_scale_d: sums += fp16(x.d @108)*y.d*(float)aux32, also
// per-super-block), then the single yield.
//
// This is the SOLE q3_K super-block lowering (the opaque monolith emitQ3_KQ8_KBlockDot
// is retired the same action as this flip). The emit is byte-identical to that
// retired monolith (same q3_K aux32 core helper, same facts, same order) modulo the
// source-op provenance token; the structural landmarks below assert the shape.
// Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. This
// is additive: it never touches the q4_K/q5_K DUAL nor the q6_K/q2_K paths (their
// branches are unchanged -- zero regression).

module {
  tcrv.exec.kernel @q3_k_super_block_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q3_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // Placeholder scratch runtime_abi_values standing in for the per-super-block
      // aux32 integer-state scratch the bricks write/read (the real scratch is a
      // function-scoped variable the loop emitter declares itself; in the loop form
      // these operand slots are vestigial, but the standalone-form C types are pinned
      // so the MGATE anti-bypass -- dropping block_index -> standalone form -- still
      // PARSES + VERIFIES before the driver gate fails legalization). %aux32out is
      // the aux32 core's int32_t* output; %aux32 is the fold brick's const int32_t*
      // aux32 slot.
      %aux32out = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "q3-aux32-scratch-out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %aux32 = tcrv_rvv.runtime_abi_value {c_name = "aux32r", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q3-aux32-scratch", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q3_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q3_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, fold_model = "scales_times_sumi"} {
        ^bb0(%super_block_index: index, %sums: !tcrv_rvv.vector<f32, "m2">):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // q3_K INTEGER CORE: the 2-bit + SUBTRACTIVE-hmask decode + the SIGNED
          // 6-bit scale dance into the per-super-block aux32[8] i32 state. The
          // `block %super_block_index` operand makes the weight/activation bases
          // per-super-block (vx + ib*110, vy + ib*292), not loop-invariant sb-0.
          %q3core = tcrv_rvv.q3_k_q8_k_aux32_partial %vx, %vy, %aux32out, %n, %vl block %super_block_index : index {kind = "ggml_q3_k_q8_k_aux32_partial", scale_model = "per-sub-block-int6-signed-scale-i32-domain", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_hmask_byte_offset = 0 : i64, weight_qs_byte_offset = 32 : i64, weight_scales_byte_offset = 96 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // REUSED no-min positive fold: sums += fp16(x.d @108)*y.d*(float)aux32 --
          // the 8-lane fp32 VECTOR sums accumulator chain, NO min term (q3_K has no
          // per-block min). weight_d_byte_offset = 108 = block_q3_K d offset.
          // Per-super-block via block_index.
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl block %super_block_index : index {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 108 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // SINGLE carried-out accumulator (sums vector ONLY -- no sumf).
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
// VERIFY: tcrv_rvv.q3_k_q8_k_aux32_partial %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.q4_k_sums_fold_scale_d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : !tcrv_rvv.vector<f32, "m2">
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The honest body lowers to a REAL emitc.func -- the byte-exact SINGLE-accumulator
// emit, byte-identical to the retired monolith emitQ3_KQ8_KBlockDot (same q3_K aux32
// core helper, same facts, same order) modulo the source-op provenance token.
// EMIT: emitc.func @tcrv_emitc_q3_k_super_block_loop_body_kernel_q3_k_super_block_loop_body(
// The q3_K-specific uint32_t utmp[4] scale-dance scratch, declared between aux8 and
// sums8 (q6_K's direct int8 scale needs no such scratch -- a discriminating landmark).
// EMIT: local_variable=aux8
// EMIT: local_variable=utmp
// EMIT: local_variable=sums8
// The SINGLE `sums` vfloat32m2 emitc.variable, seeded ONCE outside the loop.
// EMIT: local_variable=sums
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the
// aux32 core brick operands: vx + ib*110, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "110"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The q3_K integer core: the 2-bit + SUBTRACTIVE-hmask decode (the hmask high-bit
// plane loaded once per chunk, `vand_vx`/`vsll_vx`/`vor_vv` then the vsub 4 for the
// signed [-4,3] predicated decrement) followed by the SIGNED 6-bit scale dance, then
// the per-sub-block signed-scaled vwmacc into the 8-lane aux32 (mf2 chain: i16m1
// product, i32m2 accumulator).
// EMIT: callee=unpack_2bit_subtractive_hmask
// EMIT: callee=hmask_high_bit_plane
// EMIT: call_opaque "__riscv_vor_vv_u8m2"
// EMIT: call_opaque "__riscv_vsub_vx_i8m2"
// EMIT: callee=signed_scale_bit_dance
// EMIT: call_opaque "__riscv_vmv_v_x_i32m2"
// EMIT: callee=signed_scale_load
// EMIT: call_opaque "__riscv_vwmul_vv_i16m1"
// EMIT: call_opaque "__riscv_vwmacc_vx_i32m2"
// The no-min positive fold (the SEPARATE vfmul/vfadd, no fma), reading the q3_K
// weight d at byte offset 108.
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
// unpack_2bit_subtractive_hmask. This would be FALSE for the honestly-wired body
// (which emits super_block_base_y right there), so the emit provably tracks the
// q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=unpack_2bit_subtractive_hmask

// MGATE anti-bypass (the operand-driven driver gate): dropping the aux32 core's
// block_index parses + verifies clean (the standalone-form int32_t* output type is
// honored) but FAILS to legalize -- the driver requires every addressing brick's
// block_index to be the loop induction variable (region arg 0), so a body that
// would silently address super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'tcrv.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the single-accumulator super-block loop region
// is fail-closed rejected (the recursive [L-8] allowlist default is deny, naming
// the offending op) -- the q3_K aux32 core clearing the gate is NOT a hole.
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
