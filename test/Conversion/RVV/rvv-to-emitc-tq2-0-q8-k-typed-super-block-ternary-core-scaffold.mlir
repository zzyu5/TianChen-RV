// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// tq2_0 anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the ternary core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x (both
// bases fold off the WEIGHT base via the per-body memo), so a valid operand edit yields
// DIFFERENT C (not a rejection). (MGATE) dropping the ternary core's block_index still
// PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the driver
// gate requires the brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/tq2_0_q8_k_ternary_core %%vx, %%vy/tq2_0_q8_k_ternary_core %%vx, %%vx/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/tq2_0_q8_k_ternary_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/tq2_0_q8_k_ternary_core %%vx, %%vy, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// tq2_0 super-block SCALAR-accumulator TERNARY emit (the flip lowering + the emitter-inlined
// per-super-block ternary body). tq2_0 is the FIRST TQ ({-1,0,+1}) TriLM family member: its
// integer core is the FUSED 2-bit TERNARY dot (decode_model=arithmetic -- q2_K's 2-bit
// `(qs>>shift)&3` unpack over the 4 shifts {0,2,4,6} + the per-element `-1` ternary bias vsub,
// vwmacc'd DIRECTLY against the matching 32 q8 lanes into a wide i16 accumulator, ONE vwredsum
// per 32-byte chunk into the per-super-block scalar sumi -- NO grid/codebook gather). Its fold
// is the SAME SINGLE per-super-block scalar `sumf` accumulator arity as iq1_s/iq3_s (fold_model
// "scalar_delta_grid"), a single-scale scalar fold `sumf += (float)sumi * d`, d = fp16(x.d @64)
// * y.d @0, NO trailing factor. So tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model = "scalar_delta_grid" carries a SCALAR accumulator: the region entry arguments are
// just (super_block_index, sumf:f32) and the region is terminated by
// tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE. The body
// carries the NET-NEW tq2_0 FUSED 2-bit TERNARY INTEGER CORE
// (tcrv_rvv.tq2_0_q8_k_ternary_core) with a per-super-block `block %super_block_index` operand,
// producing ONE SCALAR i32 result (the per-super-block sumi placeholder -- the emitter re-emits
// the whole body including the fold). It SHARES weight_block_stride 66 with iq2_xxs but the
// DISTINCT brick op type disambiguates the emitter.
//
// THE FLIP (C_construct 24->25): the front door now constructs this typed body as the SOLE
// representation of the tq2_0 vec_dot (the monolith op + emitter + verifier were retired the
// same action). The emit is byte-identical to the retired monolith (same fused 2-bit plane
// ternary dot + single-scale scalar fp32 fold, same facts, same order) modulo the source-op
// provenance token + the func name. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv),
// not tested here. This is additive: it never touches the q4_K/q5_K DUAL, the q6_K
// SINGLE-vector, the q2_K scalar, nor the iq1_s/iq1_m/iq3_xxs/iq2_*/iq3_s/iq4_xs grid/codebook
// paths (their fold_model / brick branches are unchanged -- zero regression). tq2_0 is the
// FIRST TQ member; tq1_0 (base-3) reuses this ternary scaffold next at C2 marginal cost.

module {
  tcrv.exec.kernel @tq2_0_super_block_ternary_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @tq2_0_super_block_ternary_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @tq2_0_super_block_ternary_core, sew = 32 : i64, source_kernel = "tq2_0_super_block_ternary_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // tq2_0 FUSED 2-bit TERNARY INTEGER CORE (NET-NEW, decode_model=arithmetic): the
          // 32-byte qs chunk loaded ONCE at e8<anchor> + the 4 2-bit planes {0,2,4,6} each
          // unpacked to 32 ternary lanes (vsrl/vand -> u8->i8 reinterpret -> the per-element
          // `-1` bias via vsub) + vwmacc'd DIRECTLY against the matching 32 q8 lanes into a
          // wide i16 accumulator + ONE vwredsum per 32-byte chunk into the per-super-block
          // scalar sumi, producing ONE SCALAR i32 result. The `block %super_block_index`
          // operand makes the weight/activation bases per-super-block (vx + ib*66, vy + ib*292).
          %sumi = tcrv_rvv.tq2_0_q8_k_ternary_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_tq2_0_q8_k_ternary_core", scale_model = "ternary-2bit-fused-plane-single-fp16-scale-i32-domain", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector `sums`).
          // The byte-exact single-scale scalar fold sumf += (float)sumi * d (d = fp16(x.d @64)
          // * y.d @0) + the trailing *s = sumf (NO factor) are emitter-inlined.
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block TERNARY body composes: the loop op
// (fold_model "scalar_delta_grid") + the tq2_0 ternary integer-core brick + the one-operand
// scalar yield round-trip (the region carries the (index, f32) pair and the yield names the
// single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.tq2_0_q8_k_ternary_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact SCALAR-accumulator
// TERNARY emit, byte-identical to the retired monolith (same fused 2-bit plane ternary dot +
// single-scale scalar fp32 fold, same facts, same order) modulo the source-op provenance token
// + the func name.
// EMIT: emitc.func @tcrv_emitc_tq2_0_super_block_ternary_core_kernel_tq2_0_super_block_ternary_core(
// The super-block count nb = n / 256. The FUSED dot has NO aux8[256] scratch (the 2-bit unpack
// is consumed in-register by the per-plane vwmacc, never spilled).
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT-NOT: !emitc.array<256x!emitc.opaque<"int8_t">>
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the loop.
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the ternary
// core brick operands: vx + ib*66, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "66"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The SINGLE per-super-block integer accumulator sumi (default anchor m2 at the VLEN-universal
// floor; the gearbox refines m2->m1 at VLEN256 via a separate pass).
// EMIT: local_variable=sumi
// EMIT: !emitc.lvalue<!emitc.opaque<"int">>
// The FUSED 2-bit TERNARY dot (ggml's _vl128 lane structure): a wide i16m4 plane accumulator
// zeroed ONCE for the super-block, each 32-byte qs chunk loaded at e8m2, then 4 planes each
// unpacking 32 ternary lanes (vsrl/vand over {0,2,4,6}, u8->i8 reinterpret, the per-element
// `-1` bias via vsub in the i8 domain) and vwmacc'd DIRECTLY against the matching 32 q8 lanes
// -- the load-bearing decode `((qs>>shift)&3) - 1` fused into one widening MAC, NO vse8 spill.
// EMIT: callee=chunk_dot
// EMIT: call_opaque "__riscv_vsetvl_e16m4"
// EMIT: call_opaque "__riscv_vmv_v_x_i16m4"
// EMIT: call_opaque "__riscv_vsetvl_e8m2"
// EMIT: call_opaque "__riscv_vle8_v_u8m2"
// EMIT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// EMIT: call_opaque "__riscv_vsub_vx_i8m2"
// EMIT: call_opaque "__riscv_vle8_v_i8m2"
// EMIT: call_opaque "__riscv_vwmacc_vv_i16m4"
// tq2_0 has NO scales / min / bsums machinery -- it must NOT misroute to a q2_K or q4_K-style
// scale/min decode, and NO codebook/grid gather. NO 4-bit nibble scale extraction, NO 6-bit
// utmp/kmask bit-dance, NO bsums, NO vluxei/vrgather. NO aux8 spill.
// EMIT-NOT: bitwise_and
// EMIT-NOT: bitwise_right_shift
// EMIT-NOT: bitwise_or
// EMIT-NOT: const int16_t
// EMIT-NOT: call_opaque "__riscv_vse8_v_i8m2"
// EMIT-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT-NOT: call_opaque "__riscv_vluxei16
// EMIT-NOT: call_opaque "__riscv_vrgather
// ONE wide widening reduce for the WHOLE super-block (i16m4 -> i32m1 -> scalar), summed
// into sumi (NO per-sub-block scale multiply -- tq2_0 has no scales).
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SINGLE-SCALE SCALAR fp32 fold: dy (fp32 q8_K scale) loaded once, dx via the fp16 read
// seam (ONE read, at xb+64), d = dy*dx, then `sumf += (float)sumi * d` as ONE emitc.expression
// (a cast + a mul + an add). There is exactly ONE fp16 read (no dall/dmin pair), NO subtract
// (no min term), NO 8-lane vfcvt/vfmul.
// EMIT: callee=fold_activation_d
// EMIT: callee=fold_scale_d
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: callee=scalar_fold
// EMIT: expression
// EMIT: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The *s store (`*s = sumf`, a SEPARATE statement OUTSIDE the loop; tq2_0 applies NO trailing
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
// op) -- the tq2_0 ternary core clearing the gate is NOT a hole.
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
