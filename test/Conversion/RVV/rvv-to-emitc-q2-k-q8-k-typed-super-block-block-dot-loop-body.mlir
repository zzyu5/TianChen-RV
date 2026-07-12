// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// M-FLAT q2_K milestone-2 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not
// gate-only): (MSWAP) swapping the q2_K integer core's q8 activation base %vy ->
// %vx CHANGES the emit -- the super_block_base_y activation-base arithmetic no
// longer precedes the Region-A unpack (the core folds off the WEIGHT base now, and
// the memo collapses both bases to super_block_base_x), so a valid operand edit
// yields DIFFERENT C (not a rejection). (MGATE) dropping the integer core's
// block_index still PARSES + VERIFIES (the standalone 4-operand form) but FAILS to
// legalize -- the driver gate requires the brick's block_index to be the loop
// induction variable so the emit addresses base + ib*stride, never super-block-0.
// RUN: sed 's/q2_k_q8_k_integer_core %%vx, %%vy/q2_k_q8_k_integer_core %%vx, %%vx/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/q2_k_q8_k_integer_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/q2_k_q8_k_integer_core %%vx, %%vy, %%n, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "scalar_scale_min"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_scale_min"/fold_model = "super_block_two_level_scale_min"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// M-FLAT q2_K super-block SCALAR-accumulator emit (milestone-2: W-D byte-exact
// scalar-accumulator lowering + the shared q2_K integer core). q2_K is a
// super-block affine quant (it HAS a min term, like q4_K/q5_K) but its fold is
// STRUCTURALLY DISTINCT: q2_K accumulates a SINGLE per-super-block scalar `sumf +=
// dall*isum - dmin*summs`, where isum and summs are SCALAR integer states -- NOT
// q4_K/q5_K's 8-lane deferred VECTOR `sums` + scalar `sumf` DUAL, and NOT q6_K's
// 8-lane VECTOR `sums`-only SINGLE. So weft_rvv.typed_super_block_block_dot_loop_body
// with fold_model = "scalar_scale_min" carries a SCALAR accumulator: the region
// entry arguments are just (super_block_index, sumf:f32) and the region is
// terminated by weft_rvv.typed_super_block_block_dot_loop_yield naming the `sumf`
// SCALAR ALONE (no 8-lane `sums` vector operand). The honest target body carries
// the NET-NEW q2_K INTEGER CORE (weft_rvv.q2_k_q8_k_integer_core -- the 2-bit
// weight unpack + PLAIN uint4-nibble scale/min extraction, NO 6-bit bit-dance, +
// per-sub-block scalar i32 dot) with a per-super-block `block %super_block_index`
// operand, producing the two SCALAR integer states isum + summs; the scalar fold
// (sumf += dall*isum - dmin*summs, fp16 d@80/dmin@82) has NO separate fold brick
// (its offsets are FIXED block_q2_K constants) -- it is emitter-inlined.
//
// This is the SOLE q2_K super-block lowering (the opaque monolith
// emitQ2_KQ8_KBlockDot is retired the same action as this flip). The emit is
// byte-identical to that retired monolith (same q2_K integer core + scalar fold
// helper, same facts, same order) modulo the source-op provenance token; the
// structural landmarks below assert the W-D shape. Numerical bit-exact-vs-ggml is
// pending-hardware (ssh rvv), not tested here. This is additive: it never touches
// the q4_K/q5_K DUAL nor the q6_K SINGLE-VECTOR path (their fold_model branches are
// unchanged -- zero regression).

module {
  weft.exec.kernel @q2_k_super_block_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @q2_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q2_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q2_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_scale_min"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // q2_K INTEGER CORE (NET-NEW): the 2-bit weight unpack `(qs >> shift) & 3`
          // + PLAIN uint4-nibble scale/min (sc & 0xF is the scale, sc >> 4 is the
          // min -- NO 6-bit utmp/kmask bit-dance) + per-sub-block scalar i32 dot,
          // producing the two SCALAR integer states isum + summs. The `block
          // %super_block_index` operand makes the weight/activation bases per-super-
          // block (vx + ib*84, vy + ib*292), not loop-invariant super-block-0.
          %isum, %summs = weft_rvv.q2_k_q8_k_integer_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_q2_k_q8_k_integer_core", scale_model = "per-sub-block-uint4-scale-i32-domain-min", qk = 256 : i64, sub_block = 16 : i64, weight_block_stride = 84 : i64, activation_block_stride = 292 : i64, weight_scales_byte_offset = 0 : i64, weight_qs_byte_offset = 16 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32, i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector
          // `sums`). The byte-exact scalar fold sumf += dall*isum - dmin*summs
          // (dall = fp16(x.d @80)*y.d, dmin = fp16(x.dmin @82)*y.d) is emitter-
          // inlined (no separate fold brick).
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse and verify -- the SCALAR-accumulator super-block loop op + the one-operand
// yield round-trip (the region carries the (index, f32) pair and the yield names
// the single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: weft_rvv.q2_k_q8_k_integer_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// Milestone-2 (W-D): the honest body lowers to a REAL emitc.func -- the byte-exact
// SCALAR-accumulator emit, byte-identical to the retired monolith
// emitQ2_KQ8_KBlockDot (same q2_K integer core + scalar fold helper, same facts,
// same order) modulo the source-op provenance token.
// EMIT: emitc.func @weft_emitc_q2_k_super_block_loop_body_kernel_q2_k_super_block_loop_body(
// The int8_t aux8[256] element-ordered scratch.
// EMIT: !emitc.array<256x!emitc.opaque<"int8_t">>
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside
// the loop (NO 8-lane sums vector -- q2_K's positive term is the scalar isum).
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the
// integer core brick operands: vx + ib*84, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "84"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The 2-bit weight unpack: u8m2 load of the qs chunk, then vand/vsrl over the
// shifts {0,2,4,6}, u8->i8 reinterpret, vse8.
// EMIT: callee=unpack_2bit
// EMIT: call_opaque "__riscv_vsetvl_e8m2"
// EMIT: call_opaque "__riscv_vle8_v_u8m2"
// EMIT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT: call_opaque "__riscv_vsrl_vx_u8m2"
// EMIT: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// EMIT: call_opaque "__riscv_vse8_v_i8m2"
// The PLAIN 4-bit nibble scale/min of the direct scales[16] bytes (NO 6-bit
// utmp/kmask bit-dance -- so NO bitwise_or), then the per-sub-block widening dot
// into the scalar isum: i8m1 loads -> vwmul -> vwredsum -> vmv_x_s.
// EMIT-NOT: bitwise_or
// EMIT: bitwise_and
// EMIT: bitwise_right_shift
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SCALAR fp32 fold: dall/dmin via the fp16 read seam, then
// `sumf += dall*isum - dmin*summs` as ONE emitc.expression (two muls + a sub + an
// add). NO 8-lane vfcvt/vfmul/vfadd, NO post-loop vse32 horizontal sum.
// EMIT: callee=fold_scale_dall
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: callee=fold_scale_dmin
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: callee=scalar_fold
// EMIT: expression
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfmacc
// EMIT-NOT: call_opaque "__riscv_vfmadd
// EMIT-NOT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The *s store.
// EMIT: callee=store_s

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the integer core's q8
// base swapped %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y
// activation arithmetic no longer precedes the Region-A unpack (both bases collapse
// to super_block_base_x off the weight base), so no super_block_base_y appears
// between super_block_base_x and the first unpack_2bit. This would be FALSE for the
// honestly-wired body (which emits super_block_base_y right there), so the emit
// provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=unpack_2bit

// MGATE anti-bypass (the operand-driven driver gate): dropping the integer core's
// block_index parses + verifies clean (the standalone 4-operand form) but FAILS to
// legalize -- the driver requires the addressing brick's block_index to be the loop
// induction variable (region arg 0), so a body that would silently address
// super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block loop region is
// fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the
// offending op) -- the q2_K integer core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7).
// The SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed:
// the scalar "scalar_scale_min" path rejects an extra region arg (a vector-carrying
// dual/triple region) and a dual (second-operand) yield; the dual
// "super_block_two_level_scale_min" path rejects this scalar (index, sumf) region.
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator q2_K fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// SCALARYIELDSECOND: must NOT carry a second
