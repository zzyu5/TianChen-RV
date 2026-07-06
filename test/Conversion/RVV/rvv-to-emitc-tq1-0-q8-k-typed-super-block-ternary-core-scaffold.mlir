// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// tq1_0 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the ternary core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x (both
// bases fold off the WEIGHT base via the per-body memo), so a valid operand edit yields
// DIFFERENT C (not a rejection). (MGATE) dropping the ternary core's block_index still
// PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the driver
// gate requires the brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/tq1_0_q8_k_ternary_core %%vx, %%vy/tq1_0_q8_k_ternary_core %%vx, %%vx/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/tq1_0_q8_k_ternary_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/tq1_0_q8_k_ternary_core %%vx, %%vy, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// tq1_0 super-block SCALAR-accumulator BASE-3 TERNARY emit (the flip lowering + the
// emitter-inlined per-super-block ternary body). tq1_0 is the SECOND TQ ({-1,0,+1}) TriLM
// family member (the base-3-packed sibling of tq2_0): its integer core is the BASE-3 TERNARY
// dot (decode_model=arithmetic -- the qs main/tail + qh base-3 trit unpack `q=(uint8_t)
// (byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` into an element-ordered aux8[256], then the
// flat-256 widened i8*i8 dot vle8 i8 x q8 i8 -> vwmul i16 -> vwredsum i32 into the
// per-super-block scalar sumi -- NO grid/codebook gather, NO 2-bit field shift). Its fold is
// the SAME SINGLE per-super-block scalar `sumf` accumulator arity as tq2_0/iq1_s (fold_model
// "scalar_delta_grid"), a single-scale scalar fold `sumf += (float)sumi * d`, d = fp16(x.d
// @52) * y.d @0, NO trailing factor. So tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model = "scalar_delta_grid" carries a SCALAR accumulator: the region entry arguments
// are just (super_block_index, sumf:f32) and the region is terminated by
// tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE. The body
// carries the tq1_0 BASE-3 TERNARY INTEGER CORE (tcrv_rvv.tq1_0_q8_k_ternary_core) with a
// per-super-block `block %super_block_index` operand, producing ONE SCALAR i32 result (the
// per-super-block sumi placeholder -- the emitter re-emits the whole body including the fold).
// Its weight_block_stride 54 is UNIQUE among the scalar_delta_grid bricks, so it dispatches by
// stride; the DISTINCT base-3 brick op type also disambiguates the emitter.
//
// THE FLIP (C_construct 25->26): the front door now constructs this typed body as the SOLE
// representation of the tq1_0 vec_dot (the monolith op + emitter + verifier were retired the
// same action). It REUSES the WHOLE tq2_0 ternary scaffold at C2 marginal cost -- the SAME loop
// op, the SAME single-yield contract, the SAME selector, the SAME emitter dispatch -- differing
// ONLY in the base-3 unpack. The emit is byte-identical to the retired monolith (same base-3
// trit unpack + flat-256 integer dot + single-scale scalar fp32 fold, same facts, same order)
// modulo the source-op provenance token + the func name. Numerical bit-exact-vs-ggml is
// pending-hardware (ssh rvv), not tested here. This is additive: it never touches the
// q4_K/q5_K DUAL, the q6_K SINGLE-vector, the q2_K scalar, the tq2_0 fused-2bit path, nor the
// iq1_s/iq1_m/iq3_xxs/iq2_*/iq3_s/iq4_xs grid/codebook paths (zero regression).

module {
  tcrv.exec.kernel @tq1_0_super_block_ternary_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @tq1_0_super_block_ternary_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @tq1_0_super_block_ternary_core, sew = 32 : i64, source_kernel = "tq1_0_super_block_ternary_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // tq1_0 BASE-3 TERNARY INTEGER CORE (decode_model=arithmetic): the qs main/tail + qh
          // base-3 trit unpack (each `q=(uint8_t)(byte*pow3[l])` uint8-wrap vmul -> `((uint16_t)q
          // *3)>>8` widening vwmulu + vsrl -> narrow -> reinterpret i8 -> the per-element `-1`
          // ternary bias vadd -> vse8) decoded into an element-ordered aux8[256], then the
          // flat-256 widened i8*i8 dot (vle8 i8 x q8 i8 -> vwmul i16 -> vwredsum i32) into the
          // per-super-block scalar sumi, producing ONE SCALAR i32 result. The `block
          // %super_block_index` operand makes the weight/activation bases per-super-block (vx +
          // ib*54, vy + ib*292).
          %sumi = tcrv_rvv.tq1_0_q8_k_ternary_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_tq1_0_q8_k_ternary_core", scale_model = "ternary-base3-single-fp16-scale-i32-domain", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 48 : i64, weight_d_byte_offset = 52 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector `sums`).
          // The byte-exact single-scale scalar fold sumf += (float)sumi * d (d = fp16(x.d @52)
          // * y.d @0) + the trailing *s = sumf (NO factor) are emitter-inlined.
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block BASE-3 TERNARY body composes: the loop
// op (fold_model "scalar_delta_grid") + the tq1_0 ternary integer-core brick + the one-operand
// scalar yield round-trip (the region carries the (index, f32) pair and the yield names the
// single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.tq1_0_q8_k_ternary_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact SCALAR-accumulator
// BASE-3 TERNARY emit, byte-identical to the retired monolith (same base-3 trit unpack +
// flat-256 integer dot + single-scale scalar fp32 fold, same facts, same order) modulo the
// source-op provenance token + the func name.
// EMIT: emitc.func @tcrv_emitc_tq1_0_super_block_ternary_core_kernel_tq1_0_super_block_ternary_core(
// The super-block count nb = n / 256, and the element-ordered int8_t aux8[256] scratch (the
// base-3 unpack destination read contiguously by the dot -- tq1_0 KEEPS the aux8 round-trip,
// unlike tq2_0's fully-fused plane).
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT: !emitc.array<256x!emitc.opaque<"int8_t">>
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the loop.
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the ternary
// core brick operands: vx + ib*54, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "54"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The BASE-3 trit unpack (the load-bearing decode `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q
// *3)>>8; xi-1`): the u8m2 load of the qs chunk, the uint8 wrap vmul.vx by pow3[l] (NOT widened
// -- the mod-256 truncation IS the decode), the widening vwmulu by 3, the vsrl by 8, the
// u16->u8 narrow, the u8->i8 reinterpret, then the per-element `-1` ternary bias (vadd.vx of
// -1) BEFORE the vse8 spill.
// EMIT: callee=unpack_base3_trit
// EMIT: call_opaque "__riscv_vsetvl_e8m2"
// EMIT: call_opaque "__riscv_vle8_v_u8m2"
// EMIT: call_opaque "__riscv_vmul_vx_u8m2"
// EMIT: call_opaque "__riscv_vwmulu_vx_u16m4"
// EMIT: call_opaque "__riscv_vsrl_vx_u16m4"
// EMIT: call_opaque "__riscv_vncvt_x_x_w_u8m2"
// EMIT: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// EMIT: call_opaque "__riscv_vadd_vx_i8m2"
// EMIT: call_opaque "__riscv_vse8_v_i8m2"
// tq1_0 is BASE-3, NOT a 2-bit field shift (tq2_0) and NOT a nibble/min K-quant: NO `& 3`
// 2-bit field mask, NO fused-plane vwmacc, NO codebook/grid gather, NO 6-bit utmp/kmask
// bit-dance, NO bsums.
// EMIT-NOT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT-NOT: call_opaque "__riscv_vwmacc_vv_i16m4"
// EMIT-NOT: bitwise_and
// EMIT-NOT: bitwise_right_shift
// EMIT-NOT: const int16_t
// EMIT-NOT: call_opaque "__riscv_vluxei16
// EMIT-NOT: call_opaque "__riscv_vrgather
// The SINGLE per-super-block integer accumulator sumi, fed by the WIDE-strip flat-256 dot over
// aux8 x q8 at the default m2 anchor: vle8 i8m2 -> vwmul_vv_i16m4 -> vwredsum_vs_i16m4_i32m1 ->
// vmv_x_s (NO per-sub-block scale multiply -- tq1_0 has no scales).
// EMIT: local_variable=sumi
// EMIT: !emitc.lvalue<!emitc.opaque<"int">>
// EMIT: callee=chunk_dot
// EMIT: call_opaque "__riscv_vle8_v_i8m2"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m4"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The OLD narrow dot path (per-16-lane vwmul_i16m2 reduce) is gone at the default m2 anchor.
// EMIT-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// The SINGLE-SCALE SCALAR fp32 fold: dx via the fp16 read seam (ONE read, at xb+52, the END of
// block_tq1_0), dy (fp32 q8_K scale) loaded once, d = dx*dy, then `sumf += (float)sumi * d` as
// ONE emitc.expression (a cast + a mul + an add). Exactly ONE fp16 read (no dall/dmin pair), NO
// subtract (no min term), NO 8-lane vfcvt/vfmul.
// EMIT: callee=fold_scale_d
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: callee=fold_activation_d
// EMIT: callee=scalar_fold
// EMIT: expression
// EMIT: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The *s store (`*s = sumf`, a SEPARATE statement OUTSIDE the loop; tq1_0 applies NO trailing
// factor, so the store_s step comment is NOT followed by a factor literal/mul).
// EMIT: callee=store_s
// EMIT-NOT: literal "0.25f"
// EMIT-NOT: literal "0.125f"

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the ternary core's q8 base swapped
// %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation arithmetic collapses
// (both bases fold to super_block_base_x off the weight base via the per-body memo), so no
// super_block_base_y appears between super_block_base_x and the sumi accumulator. This would be
// FALSE for the honestly-wired body (which emits super_block_base_y right there), so the emit
// provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: local_variable=sumi

// MGATE anti-bypass (the operand-driven driver gate): dropping the ternary core's block_index
// parses + verifies clean (the standalone 4-operand form) but FAILS to legalize -- the driver
// requires the addressing brick's block_index to be the loop induction variable (region arg
// 0), so a body that would silently address super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'tcrv.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block TERNARY loop region is
// fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the offending
// op) -- the tq1_0 ternary core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7). The
// SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed: the scalar
// "scalar_delta_grid" path rejects an extra region arg (a vector-carrying dual/triple region)
// and a dual (second-operand) yield; the dual "super_block_two_level_scale_min" path rejects
// this scalar (index, sumf) region.
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator iq1_s fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// SCALARYIELDSECOND: must NOT carry a second
