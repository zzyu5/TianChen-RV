// RUN: tcrv-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// iq3_s anti-bypass M-tests (the emit is OPERAND-DRIVEN, not gate-only): (MSWAP)
// swapping the grid core's q8 activation base %vy -> %vx CHANGES the emit -- the
// super_block_base_y activation-base arithmetic collapses into super_block_base_x (both
// bases fold off the WEIGHT base via the per-body memo), so a valid operand edit yields
// DIFFERENT C (not a rejection). (MGATE) dropping the grid core's block_index still
// PARSES + VERIFIES (the standalone 4-operand form) but FAILS to legalize -- the driver
// gate requires the brick's block_index to be the loop induction variable so the emit
// addresses base + ib*stride, never super-block-0.
// RUN: sed 's/iq3_s_q8_k_grid_core %%vx, %%vy/iq3_s_q8_k_grid_core %%vx, %%vx/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MSWAP
// RUN: sed 's/iq3_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl block %%super_block_index : index/iq3_s_q8_k_grid_core %%vx, %%vy, %%n, %%vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "unsupported_fold"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%sumf: f32):/%%sumf: f32, %%extra: f32):/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADARGS-SCALAR
// RUN: sed 's/fold_model = "scalar_delta_grid"/fold_model = "super_block_two_level_scale_min"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=DUALFOLD
// RUN: sed 's/kind = "typed_super_block_block_dot_loop_body"/kind = "plain_super_block_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed -e 's|// RS ||' -e 's/loop_yield %%sumf : f32/loop_yield %%sumf, %%spare_sumf : f32, f32/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=SCALARYIELDSECOND

// iq3_s super-block SCALAR-accumulator GRID-of-4 EXPLICIT-SIGNS emit (the flip lowering +
// the shared per-super-block grid body). iq3_s is a super-block quant whose sub-block
// decode is a codebook GRID GATHER (decode_model=lookup, the 512-entry uint32 iq3s_grid
// GRID-of-4 gathered via vluxei16_v_i32m1), the iq3_xxs GRID-of-4 sibling with THREE
// mechanisms swapped to iq2_s: the qh 9th-bit inject (a SINGLE bit, mask 256, the two
// passes taking shifts 8-2l / 7-2l), the per-lane sign read from an EXPLICIT per-sub-block
// signs region (at offset 74, NO ksigns plane), and the per-sub-block scale from the
// EXPLICIT two-nibble scales[] (at offset 106). Its fold is a SINGLE per-super-block scalar
// `sumf += d*(float)bsum` with the trailing `*s = sumf` (NO 1/4 or 1/8 factor -- iq3_s
// applies none), where bsum (the per-sub-block 4-bit-scaled grid/sign integer dot) is a
// SCALAR integer state -- the SAME scalar-accumulator arity as iq1_s's "scalar_delta_grid",
// with a DISTINCT integer core. So tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model = "scalar_delta_grid" carries a SCALAR accumulator: the region entry arguments
// are just (super_block_index, sumf:f32) and the region is terminated by
// tcrv_rvv.typed_super_block_block_dot_loop_yield naming the `sumf` SCALAR ALONE. The body
// carries the NET-NEW iq3_s GRID-of-4 explicit-signs INTEGER CORE
// (tcrv_rvv.iq3_s_q8_k_grid_core -- the explicit two-nibble scale + qh 9th-bit inject +
// explicit per-sub-block signs decode + the two-index-per-group vluxei16_v_i32m1 grid gather
// + the signed widening grid dot + the bsum fold) with a per-super-block
// `block %super_block_index` operand, producing the ONE SCALAR integer state bsum.
//
// THE FLIP (C_construct 22->23): the front door now constructs this typed body as the SOLE
// representation of the iq3_s vec_dot (the monolith op + emitter + verifier were retired the
// same action). The emit is byte-identical to the retired monolith (same grid/kmask decls +
// per-super-block grid body helper, same facts, same order) modulo the source-op provenance
// token + the func name. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not
// tested here. This is additive: it never touches the q4_K/q5_K DUAL, the q6_K SINGLE-vector,
// the q2_K scalar, nor the iq1_s/iq1_m/iq3_xxs/iq2_* grid paths (their fold_model / brick
// branches are unchanged -- zero regression).

module {
  tcrv.exec.kernel @iq3_s_super_block_grid_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @iq3_s_super_block_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq3_s_super_block_grid_core, sew = 32 : i64, source_kernel = "iq3_s_super_block_grid_core_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // RS %spare_sumf = arith.constant 0.000000e+00 : f32
          // iq3_s GRID-of-4 explicit-signs INTEGER CORE (NET-NEW, decode_model=lookup): the
          // EXPLICIT two-nibble scale `ls = 2*(sc&0xf)+1` (even) / `2*(sc>>4)+1` (odd), the
          // qh byte, the 4 sign groups each reading the EXPLICIT sign byte sgn[ib*4+l] +
          // two qh-9th-bit-injected grid INDICES `qs[l] | ((qh<<(8-2l))&256)`, the
          // vluxei16_v_i32m1 gather over the 512-entry uint32 iq3s_grid, the signed widening
          // grid dot, and `bsum += sumi*ls`, producing the ONE SCALAR integer state bsum. The
          // `block %super_block_index` operand makes the weight/activation bases
          // per-super-block (vx + ib*110, vy + ib*292).
          %bsum = tcrv_rvv.iq3_s_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq3_s_q8_k_grid_core", scale_model = "per-sub-block-explicit-scale-grid-of-4-codebook-qh-plane-explicit-signs-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, weight_qh_byte_offset = 66 : i64, weight_signs_byte_offset = 74 : i64, weight_scales_byte_offset = 106 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (sumf ONLY -- no 8-lane vector `sums`).
          // The byte-exact scalar fold sumf += d*(float)bsum (d = fp16(x.d @0)*y.d @0) + the
          // trailing *s = sumf (NO factor) are emitter-inlined (no fold brick).
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// Parse + verify -- the SCALAR-accumulator super-block GRID-of-4 body composes: the loop op
// (fold_model "scalar_delta_grid") + the iq3_s grid-of-4 integer core brick + the one-operand
// scalar yield round-trip (the region carries the (index, f32) pair and the yield names the
// single carried-out `sumf` scalar, NO 8-lane `sums` vector).
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: tcrv_rvv.iq3_s_q8_k_grid_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}} : f32
// VERIFY-NOT: tcrv_rvv.typed_super_block_block_dot_loop_yield %{{.*}}, %{{.*}}

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact SCALAR-accumulator
// GRID-of-4 emit, byte-identical to the retired monolith (same grid/kmask decls +
// per-super-block grid body helper, same facts, same order) modulo the source-op provenance
// token + the func name.
// EMIT: emitc.func @tcrv_emitc_iq3_s_super_block_grid_core_kernel_iq3_s_super_block_grid_core(
// The 512-entry GRID-of-4 codebook + the inline kmask, emitted ONCE as structured static
// const decls (keyed off the grid-core brick op identity from the canonical kIQ3SGrid).
// iq3_s has NO ksigns plane (the signs are an explicit memory region).
// EMIT: verbatim "static const uint32_t tcrv_iq3s_grid[512] = {0x01010101U,
// EMIT: verbatim "static const uint8_t tcrv_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
// EMIT-NOT: tcrv_iq3s_ksigns
// The SINGLE `sumf` float emitc.variable SCALAR accumulator, seeded ONCE outside the loop
// (NO 8-lane sums vector -- iq3_s's positive term is the scalar bsum).
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: callee=super_block_count
// EMIT: div %{{.*}}, %{{.*}} :
// The 8-lane kmask load + the (const int32_t *) grid32 view, ONCE above the loop.
// EMIT: call_opaque "__riscv_vle8_v_u8m1"
// EMIT: callee=grid_table_i32_view
// EMIT: callee=super_block_loop
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Per-super-block addressing (x weight-base then y activation-base, built from the grid core
// brick operands: vx + ib*110, vy + ib*292 -- not super-block-0).
// EMIT: callee=super_block_base_x
// EMIT: literal "110"
// EMIT: callee=super_block_base_y
// EMIT: literal "292"
// The per-super-block fp16*fp32 scale d (the fp16 weight d read seam + the fp32 y.d).
// EMIT: call_opaque "(float)*(const _Float16 *)"
// The ONE int32 SCALAR accumulator bsum (NO aux32 memory state, NO 8-lane vector).
// EMIT: local_variable=bsum
// EMIT: !emitc.lvalue<!emitc.opaque<"int32_t">>
// The EXPLICIT two-nibble scale (`ls = 2*(sc&0xf)+1` even / `2*(sc>>4)+1` odd) + the qh
// 9th-bit inject (`(qh<<(8-2l))&256` / `(qh<<(7-2l))&256` bitwise or'd into the grid index).
// EMIT: bitwise_and
// EMIT: bitwise_left_shift
// EMIT: bitwise_or
// EMIT: bitwise_right_shift
// The HARDWARE vluxei16 grid-of-4 gather of the qh-injected indices, BATCHED per SUB-BLOCK
// PAIR (the AVL=2 fractional-LMUL qh fix, iq3_s's heaviest gap): the 32 per-group vl=2
// vluxei16_v_i32m1 gathers (each with a vl=2 u16mf2 index load -- the 2-element
// scalarization storm) are HOISTED to ONE wide gather per pair (2 sub-blocks = 8 groups =
// 16 qh-injected indices). A uint16_t[32] byte-offset array laid out 4 slots/group (2 real
// idx*4 + 2 zero pads) is vle16'd at u16m4, gathered by ONE vluxei16_v_i32m8 over the i32
// grid base, reinterpreted to i8m8; each group's 8 signed grid bytes are recovered in
// lanes 0..7 by a register-group vget (i8m8 -> i8m1) and fed to the UNCHANGED EXPLICIT-signs
// fold (broadcast signs / vand kmask / vmsne / vneg / vmerge) + signed widening dot + ONE
// vwredsum per group. The qh injection and the per-group integer dot are byte-identical
// (the pads fill the unread lanes 8..15); only the gather + vsetvli config are hoisted.
// EMIT: "emitc.variable"() {{.*}} -> !emitc.array<32x!emitc.opaque<"uint16_t">>
// EMIT: call_opaque "__riscv_vle16_v_u16m4"
// EMIT: call_opaque "__riscv_vluxei16_v_i32m8"
// EMIT: call_opaque "__riscv_vreinterpret_v_i32m8_i8m8"
// EMIT: call_opaque "__riscv_vget_v_i8m8_i8m1"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vmv_v_x_u8m1"
// EMIT: call_opaque "__riscv_vand_vv_u8m1"
// EMIT: call_opaque "__riscv_vmsne_vx_u8m1_b8"
// EMIT: call_opaque "__riscv_vneg_v_i8m1"
// EMIT: call_opaque "__riscv_vmerge_vvm_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The per-group vl=2 fractional-LMUL grid gather is FULLY ELIMINATED (no i32m1 gather,
// no u16mf2 index load remain -- the AVL=2 scalarization storm is gone).
// EMIT-NOT: call_opaque "__riscv_vluxei16_v_i32m1"
// EMIT-NOT: call_opaque "__riscv_vle16_v_u16mf2"
// EMIT-NOT: call_opaque "__riscv_vreinterpret_v_i32m1_i8m1"
// The per-sub-block bsum accumulate `bsum += sumi * ls` (integer, order-free).
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fp32 fold `sumf += d*(float)bsum` as ONE emitc.expression. NO 8-lane
// vfcvt/vfmul, NO post-loop vse32 horizontal sum.
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT-NOT: call_opaque "__riscv_vfredusum
// The *s store (`*s = sumf`, a SEPARATE statement OUTSIDE the loop; iq3_s applies NO trailing
// 1/4 or 1/8 factor, so the store_s step comment is NOT followed by a factor literal/mul).
// EMIT: callee=store_s
// EMIT-NOT: literal "0.25f"
// EMIT-NOT: literal "0.125f"

// MSWAP anti-bypass (operand-driven, NOT gate-only): with the grid core's q8 base swapped
// %vy -> %vx, the emit is DIFFERENT -- the super_block_base_y activation arithmetic collapses
// (both bases fold to super_block_base_x off the weight base via the per-body memo), so no
// super_block_base_y appears between super_block_base_x and the fp16 scale read. This would be
// FALSE for the honestly-wired body (which emits super_block_base_y right there), so the emit
// provably tracks the q8_base OPERAND.
// MSWAP: callee=super_block_base_x
// MSWAP-NOT: callee=super_block_base_y
// MSWAP: callee=fcvt.s.h

// MGATE anti-bypass (the operand-driven driver gate): dropping the grid core's block_index
// parses + verifies clean (the standalone 4-operand form) but FAILS to legalize -- the driver
// requires the addressing brick's block_index to be the loop induction variable (region arg
// 0), so a body that would silently address super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'tcrv.exec.variant'
// MGATE-NOT: allowlist

// A TOP-LEVEL non-allowlist op in the scalar-accumulator super-block GRID loop region is
// fail-closed rejected (the recursive [L-8] allowlist default is deny, naming the offending
// op) -- the iq3_s grid core clearing the gate is NOT a hole.
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
