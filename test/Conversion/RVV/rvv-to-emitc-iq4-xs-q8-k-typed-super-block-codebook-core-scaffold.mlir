// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// iq4_xs anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the codebook core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x (both
// bases fold off the WEIGHT base via the per-body memo), so a valid operand edit yields
// DIFFERENT C (not a rejection). (MGATE) dropping the codebook core's block_index still
// PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the driver
// gate requires the brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/iq4_xs_q8_k_codebook_core %%vx, %%vy/iq4_xs_q8_k_codebook_core %%vx, %%vx/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/iq4_xs_q8_k_codebook_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/iq4_xs_q8_k_codebook_core %%vx, %%vy, %%n, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND
// RUN: sed 's/-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113/-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADCODEBOOK

// iq4_xs super-block SCALAR-accumulator CODEBOOK emit (the flip lowering + the emitter-inlined
// per-super-block codebook body). iq4_xs is the SUPER-BLOCK rung of the flat iq4_nl codebook:
// its sub-block decode is the SAME 16-entry non-linear int8 CODEBOOK GATHER (decode_model=
// lookup, the kvalues_iq4nl[16] table gathered via vrgather_vv_i8m1) iq4_nl uses, wrapped in
// the q4_K-style super-block SIGNED 6-bit scale machinery (per-sub-block scale from
// scales_l[4] + scales_h, biased -32, applied in the FLOAT domain -- NO integer aux32). Its
// fold is the SAME SINGLE per-super-block scalar `sumf` accumulator arity as iq1_s/iq3_s
// (fold_model "scalar_delta_grid"), but UNLIKE the grid siblings it runs PER-SUB-BLOCK in
// float `sumf += (d4d8*(ls-32))*(float)sumi` (8 fp folds per super-block, NO trailing factor).
// So weft_rvv.typed_super_block_block_dot_loop_body with fold_model = "scalar_delta_grid"
// carries a SCALAR accumulator: the region entry arguments are just (super_block_index,
// sumf:f32) and the region is terminated by weft_rvv.typed_super_block_block_dot_loop_yield
// naming the `sumf` SCALAR ALONE. The body carries the NET-NEW iq4_xs CODEBOOK INTEGER CORE
// (weft_rvv.iq4_xs_q8_k_codebook_core -- the 16-entry vrgather codebook gather + the q4_K-style
// signed 6-bit scale bit-dance + the per-sub-block float fold) with a per-super-block
// `block %super_block_index` operand, producing ONE SCALAR i32 result (an UNUSED placeholder --
// iq4_xs's fold has no single scalar state).
//
// THE FLIP (C_construct 23->24): the front door now constructs this typed body as the SOLE
// representation of the iq4_xs vec_dot (the monolith op + emitter + verifier were retired the
// same action). The emit is byte-identical to the retired monolith (same codebook decl +
// per-super-block body, same facts, same order) modulo the source-op provenance token + the
// func name. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. This
// is additive: it never touches the q4_K/q5_K DUAL, the q6_K SINGLE-vector, the q2_K scalar,
// nor the iq1_s/iq1_m/iq3_xxs/iq2_*/iq3_s grid paths (their fold_model / brick branches are
// unchanged -- zero regression). iq4_xs is the FIRST super-block CODEBOOK member (the prior
// scalar_delta_grid members are all GRID codebooks; iq4_xs is the FLAT-codebook sibling).

module {
  weft.exec.kernel @iq4_xs_super_block_codebook_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @iq4_xs_super_block_codebook_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq4_xs_super_block_codebook_core, sew = 32 : i64, source_kernel = "iq4_xs_super_block_codebook_core_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // iq4_xs CODEBOOK INTEGER CORE (NET-NEW, decode_model=lookup): the 16-entry
          // non-linear int8 codebook (kvalues_iq4nl[16]) broadcast ONCE + the per-sub-block
          // SIGNED 6-bit scale bit-dance (ls = ((scales_l[j/2]>>(4*(j%2)))&0xf) |
          // (((scales_h>>(2*j))&0x3)<<4), biased -32) + the codebook dot (vand/vsrl nibble
          // split -> two vrgather_vv_i8m1 gathers -> asymmetric vwmul/vwmacc widening product
          // -> seed-0 vwredsum) + the per-sub-block float fold `sumf += (d4d8*(ls-32))*sumi`,
          // producing ONE SCALAR i32 result (UNUSED -- the fold is per-sub-block float, no
          // single scalar state). The `block %super_block_index` operand makes the
          // weight/activation bases per-super-block (vx + ib*136, vy + ib*292).
          %partial = weft_rvv.iq4_xs_q8_k_codebook_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq4_xs_q8_k_codebook_core", scale_model = "per-sub-block-signed-6bit-scale-codebook-gather-float-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_scales_h_byte_offset = 2 : i64, weight_scales_l_byte_offset = 4 : i64, weight_qs_byte_offset = 8 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector `sums`).
          // The byte-exact per-sub-block float fold sumf += (d4d8*(ls-32))*(float)sumi (d4d8 =
          // fp16(x.d @0)*y.d @0) + the trailing *s = sumf (NO factor) are emitter-inlined.
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block CODEBOOK body composes: the loop op
// (fold_model "scalar_delta_grid") + the iq4_xs codebook integer-core brick + the one-operand
// scalar yield round-trip (the region carries the (index, f32) pair and the yield names the
// single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: weft_rvv.iq4_xs_q8_k_codebook_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: weft_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact SCALAR-accumulator
// CODEBOOK emit, byte-identical to the retired monolith (same codebook decl + per-super-block
// body, same facts, same order) modulo the source-op provenance token + the func name.
// EMIT: emitc.func @weft_emitc_iq4_xs_super_block_codebook_core_kernel_iq4_xs_super_block_codebook_core(
// The 16-entry non-linear int8 codebook, emitted ONCE as a structured static const decl (the
// SAME kvalues_iq4nl[16] table iq4_nl uses), broadcast-loaded ONCE (vle8_v_i8m1) above the loop.
// EMIT: verbatim "static const int8_t weft_iq4_xs_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the loop.
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT: callee=codebook_table_load
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the codebook
// core brick operands: vx + ib*136, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "136"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The per-super-block d4d8 = fp16(x.d @0) * fp32(y.d @0) (the fp16 weight d read seam + the
// fp32 y.d), computed ONCE.
// EMIT: call_opaque "(float)*(const _Float16 *)"
// EMIT: callee=fold_scale_d4d8
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The uint16 scales_h load @2.
// EMIT: callee=scales_h_load
// The SIGNED 6-bit scale extraction (`ls = ((scales_l>>...)&0xf)|(((scales_h>>...)&0x3)<<4)`,
// then `ls - 32`), all scalar STRUCTURED emitc bitwise ops.
// EMIT: callee=sub_block_signed_scale
// EMIT: bitwise_right_shift
// EMIT: bitwise_and
// EMIT: bitwise_left_shift
// EMIT: bitwise_or
// EMIT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// d1 = d4d8 * (float)(ls - 32): a SEPARATE cast + mul (NOT fused with sumi).
// EMIT: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// The per-sub-block CODEBOOK dot: vsetvl_e8m1(16) + the two vrgather codebook gathers (NOT an
// arithmetic nibble-8 decode) + the asymmetric widening product + seed-0 vwredsum + extract.
// EMIT: callee=sub_block_codebook_dot
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vle8_v_u8m1"
// EMIT: call_opaque "__riscv_vand_vx_u8m1"
// EMIT: call_opaque "__riscv_vsrl_vx_u8m1"
// EMIT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwmacc_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The per-sub-block fp32 fold `sumf += d1 * (float)sumi` as ONE emitc.expression (iq4_xs is
// SYMMETRIC -- NO min term, NO integer-domain aux32). NO 8-lane vfcvt/vfmul, NO horizontal sum.
// EMIT: callee=fp32_accumulate
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT-NOT: call_opaque "__riscv_vfredusum
// EMIT-NOT: call_opaque "__riscv_vluxei16
// The *s store (`*s = sumf`, a SEPARATE statement OUTSIDE the loop; iq4_xs applies NO trailing
// factor, so the store_s step comment is NOT followed by a factor literal/mul).
// EMIT: callee=store_s
// EMIT-NOT: literal "0.25f"
// EMIT-NOT: literal "0.125f"

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the codebook core's q8 base swapped
// %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation arithmetic collapses
// (both bases fold to super_block_base_x off the weight base via the per-body memo), so no
// super_block_base_y appears between super_block_base_x and the fp16 scale read. This would be
// FALSE for the honestly-wired body (which emits super_block_base_y right there), so the emit
// provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=fcvt.s.h

// MGATE anti-bypass (the operand-driven driver gate): dropping the codebook core's block_index
// parses + verifies clean (the standalone 4-operand form) but FAILS to legalize -- the driver
// requires the addressing brick's block_index to be the loop induction variable (region arg
// 0), so a body that would silently address super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block CODEBOOK loop region is
// fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the offending
// op) -- the iq4_xs codebook core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed super-block block-dot loop-body allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7). The
// SCALAR-accumulator region/yield contract is fail-closed and fold_model-keyed: the scalar
// "scalar_delta_grid" path rejects an extra region arg (a vector-carrying dual/triple region)
// and a dual (second-operand) yield; the dual "super_block_two_level_scale_min" path rejects
// this scalar (index, sumf) region. The codebook-core brick is fail-closed on its 16-entry
// codebook fact.
// BADFOLD: currently supports only fold_model "super_block_two_level_scale_min"
// BADARGS-SCALAR: requires the region to carry exactly two entry arguments for the scalar-accumulator iq1_s fold_model
// DUALFOLD: requires the region to carry exactly three entry arguments
// BADKIND: currently supports only kind "typed_super_block_block_dot_loop_body"
// SCALARYIELDSECOND: must NOT carry a second
// BADCODEBOOK: requires codebook to carry exactly 16 int8 entries
