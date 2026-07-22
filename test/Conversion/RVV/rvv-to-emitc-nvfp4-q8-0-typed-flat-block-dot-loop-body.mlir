// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// nvfp4 anti-bypass M-test (MGATE, the operand-driven driver gate): dropping the
// codebook core's block_index still PARSES + VERIFIES (the standalone 4-operand
// form) but FAILS to legalize -- the driver gate requires the brick's block_index to
// be the loop induction variable so the emit addresses base + ib*stride, never
// super-block-0.
// RUN: sed 's/nvfp4_q8_0_codebook_core %%vx, %%vy, %%n, %%vl block %%block_index : index/nvfp4_q8_0_codebook_core %%vx, %%vy, %%n, %%vl/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MGATE
// RUN: sed 's/fold_model = "flat_nvfp4_codebook"/fold_model = "unsupported_fold"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/%%block_index: index, %%acc: f32/%%block_index: index, %%acc: f32, %%extra: f32/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADARGS
// RUN: sed 's/kind = "typed_flat_block_dot_loop_body"/kind = "plain_flat_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12/0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADCODEBOOK
// The codebook anchor is a VLMAX >= codebook.size() CAPABILITY predicate, NOT a
// hardcoded `== "m1"` literal: at the byte-exact VLEN128 a narrow mf2 anchor's gather
// VLMAX is 8 < 16, so it is rejected fail-closed WITH THE VLMAX REASON (NARROWANCHOR);
// the m1 anchor (VLMAX 16) verifies clean and round-trips the attr (WIDEANCHOR).
// RUN: sed 's/integer_core_lmul = "m1"/integer_core_lmul = "mf2"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=NARROWANCHOR
// RUN: weft-opt %s | FileCheck %s --check-prefix=WIDEANCHOR

// nvfp4 (NVIDIA's FP4, the SECOND FP4-CODEBOOK sibling) constructed FLAT-loop emit
// (the flip lowering + the emitter-inlined per-super-block codebook body). nvfp4 is a
// SUPER-BLOCK codebook quant (block_nvfp4 = {uint8_t d[4]; uint8_t qs[32]}, QK=64,
// four 16-element sub-blocks) whose 64 elements span TWO block_q8_0 activation blocks
// -- a FLAT block_q8_0 stream (like q1_0), so it rides the FLAT loop op
// (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook"), NOT the q8_K
// super-block one. Its sub-block decode is the SAME 16-entry DOUBLED e2m1 CODEBOOK
// GATHER (decode_model=lookup, kvalues_mxfp4[16] via vrgather_vv_i8m1) mxfp4 uses,
// wrapped in the per-SUB-block UE4M3 fp8 weight scale (the ldexpf-based HALF-form
// decode) + the two-q8_0-block/half addressing. Its fold is a SINGLE per-super-block
// scalar `acc` accumulator: the region entry arguments are (block_index, acc:f32) and
// the region is terminated by weft_rvv.typed_flat_block_dot_loop_yield naming the
// loop-carried `acc` (the per-sub-block fp32 fold is emitter-inlined). The body
// carries the NET-NEW nvfp4 CODEBOOK INTEGER CORE
// (weft_rvv.nvfp4_q8_0_codebook_core) with a per-super-block `block %block_index`
// operand, producing ONE SCALAR i32 result (an UNUSED placeholder -- nvfp4's fold has
// no single scalar state).
//
// THE FLIP (C_construct 27->28, the LAST dispatch-wired vec_dot): the front door now
// constructs this typed body as the SOLE representation of the nvfp4 vec_dot (the
// monolith op + emitter + verifier were retired the same action). The emit is
// byte-identical to the retired monolith (same codebook decl + per-super-block body,
// same facts, same order) modulo the source-op provenance token + the func name.
// Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. This is
// additive: it never touches q1_0's binary-sign flat body, iq4_nl's decomposed
// codebook flat body, nor mxfp4 (the FP4-class negative control, still a monolith) --
// zero regression.

module {
  weft.exec.kernel @nvfp4_flat_codebook_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @nvfp4_flat_codebook_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @nvfp4_flat_codebook_core, sew = 32 : i64, source_kernel = "nvfp4_flat_codebook_core_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 64 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, fold_model = "flat_nvfp4_codebook"} {
        ^bb0(%block_index: index, %acc: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // nvfp4 CODEBOOK INTEGER CORE (NET-NEW, decode_model=lookup): the 16-entry
          // DOUBLED e2m1 codebook (kvalues_mxfp4[16]) broadcast ONCE + the four
          // UE4M3-scaled 16-element sub-blocks' codebook dot (vand/vsrl nibble split ->
          // two vrgather_vv_i8m1 gathers -> asymmetric vwmul/vwmacc widening product ->
          // seed-0 vwredsum) + the per-sub-block UE4M3 fp8 weight scale (ldexpf HALF
          // form) + the per-sub-block float fold `acc += (dy*d)*sumi`, producing ONE
          // SCALAR i32 result (UNUSED -- the fold is per-sub-block float, no single
          // scalar state). The `block %block_index` operand makes the weight base
          // per-super-block (vx + ib*36) and the q8 block-pair base (2*ib).
          %partial = weft_rvv.nvfp4_q8_0_codebook_core %vx, %vy, %n, %vl block %block_index : index {kind = "ggml_nvfp4_q8_0_codebook_core", scale_model = "ue4m3-half-per-sub-block", qk = 64 : i64, qk_sub = 16 : i64, weight_block_stride = 36 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 4 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 8 : i64, integer_core_lmul = "m1", codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          // SINGLE carried-out SCALAR accumulator (acc ONLY). The byte-exact
          // per-sub-block float fold acc += (dy*d)*(float)sumi + the trailing *s = acc
          // (NO factor) are emitter-inlined.
          weft_rvv.typed_flat_block_dot_loop_yield %acc : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse + verify -- the FLAT-loop CODEBOOK body composes: the loop op (fold_model
// "flat_nvfp4_codebook") + the nvfp4 codebook integer-core brick + the one-operand
// scalar yield round-trip (the region carries the (index, f32) pair and the yield
// names the single carried-out `acc`).
// VERIFY: weft_rvv.typed_flat_block_dot_loop_body
// VERIFY: ^bb0(%{{.*}}: index, %{{.*}}: f32):
// VERIFY: weft_rvv.nvfp4_q8_0_codebook_core %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} block %{{.*}}
// VERIFY: weft_rvv.typed_flat_block_dot_loop_yield %{{.*}} : f32

// The flip lowers the honest body to a REAL emitc.func -- the byte-exact FLAT-loop
// CODEBOOK emit, byte-identical to the retired monolith (same codebook decl +
// per-super-block body, same facts, same order) modulo the source-op provenance token
// + the func name. The <math.h> include is added for the UE4M3 ldexpf.
// EMIT: emitc.include <"math.h">
// EMIT: emitc.func @weft_emitc_nvfp4_flat_codebook_core_kernel_nvfp4_flat_codebook_core(
// The 16-entry DOUBLED e2m1 codebook, emitted ONCE as a structured static const decl
// (the SAME kvalues_mxfp4[16] table mxfp4 uses), broadcast-loaded ONCE (vle8_v_i8m1).
// EMIT: verbatim "static const int8_t weft_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};"
// The SINGLE `sumf` float emitc.variable SCALAR accumulator + the SUPER-block count
// nb = n / 64.
// EMIT: local_variable=sumf
// EMIT: !emitc.lvalue<!emitc.opaque<"float">>
// EMIT: callee=block_count
// EMIT: div %{{.*}}, %{{.*}} :
// EMIT: callee=codebook_table_load
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// The outer super-block loop + the per-super-block weight base (vx + ib*36) + the q8
// block-pair base (2*ib), built from the codebook-core brick operands (not super-block-0).
// EMIT: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// EMIT: callee=super_block_base_x
// EMIT: literal "36"
// EMIT: callee=q8_block_base_index
// The STRUCTURED UE4M3 -> fp32 weight scale (the genuinely-new NVFP4-class piece): the
// exp/man split, the two ldexpf branches, the exp==0 conditional, the *0.5f, then the
// two specials (e==0 || e==0x7F -> 0.0f via logical_or). NOT mxfp4's E8M0 bit dance.
// EMIT: callee=ue4m3_scale_load
// EMIT: callee=ue4m3_exp_man_split
// EMIT: bitwise_right_shift
// EMIT: bitwise_and
// EMIT: callee=ue4m3_ldexpf_branches
// EMIT: call_opaque "ldexpf"
// EMIT: call_opaque "ldexpf"
// EMIT: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// EMIT: callee=ue4m3_specials
// EMIT: logical_or %{{.*}}, %{{.*}}
// EMIT: conditional %{{.*}}, %{{.*}}, %{{.*}} : !emitc.opaque<"float">
// EMIT-NOT: literal "0x00200000"
// The q8_0 fp16->fp32 scale read.
// EMIT: call_opaque "(float)*(const _Float16 *)"
// The per-sub-block CODEBOOK dot: vsetvl_e8m1(8) + the packed FP4 u8 load + the two q8
// signed halves + the two vrgather codebook gathers (NOT an arithmetic nibble-8 decode)
// + the asymmetric widening product + seed-0 vwredsum + extract.
// EMIT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT: call_opaque "__riscv_vle8_v_u8m1"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vand_vx_u8m1"
// EMIT: call_opaque "__riscv_vsrl_vx_u8m1"
// EMIT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT: call_opaque "__riscv_vwmul_vv_i16m2"
// EMIT: call_opaque "__riscv_vwmacc_vv_i16m2"
// EMIT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMIT: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook, not nibble-8).
// EMIT-NOT: call_opaque "__riscv_vxor_vx_i8
// The per-sub-block fp32 fold `sumf += (dy*d)*(float)sumi` as ONE emitc.expression, dy*d
// FIRST (ggml nvfp4 scales-first order). NO trailing factor.
// EMIT: callee=fp32_accumulate
// EMIT: expression : !emitc.opaque<"float">
// EMIT: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// EMIT: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store (`*s = sumf`, OUTSIDE the loop; nvfp4 applies NO trailing factor).
// EMIT: callee=store_s
// EMIT-NOT: literal "0.25f"
// EMIT-NOT: literal "0.125f"

// A TOP-LEVEL non-allowlist op in the FLAT CODEBOOK loop region is fail-closed
// rejected (the recursive [L-8] allowlist default is deny, naming the offending op) --
// the nvfp4 codebook core clearing the gate is NOT a hole.
// REJECT: 'arith.constant' op is not in the M-FLAT typed flat block-dot loop-body allowlist

// MGATE anti-bypass (the operand-driven driver gate): dropping the codebook core's
// block_index parses + verifies clean (the standalone 4-operand form) but FAILS to
// legalize -- the driver requires the addressing brick's block_index to be the loop
// induction variable (region arg 0), so a body that would silently address
// super-block-0 is fail-closed rejected.
// MGATE: failed to legalize operation 'weft.exec.variant'
// MGATE-NOT: allowlist

// The bounded surface is fail-closed on the loop kind and the fold_model fact (I7),
// the region arg arity, and the brick's 16-entry codebook fact.
// BADFOLD: currently supports only fold_model
// BADARGS: requires the region to carry exactly two entry arguments
// BADKIND: currently supports only kind "typed_flat_block_dot_loop_body"
// BADCODEBOOK: requires codebook to carry exactly 16 int8 entries

// The codebook anchor de-lottery: the verifier gates the anchor on the VLMAX >=
// codebook.size() CAPABILITY fact (the SAME getRVVStripVLMAXElements truth source the
// codebook emitter's getRVVCodebookGatherAnchorLMUL formula selects with), NOT a
// literal `== "m1"`. mf2's gather VLMAX 8 < 16 at VLEN128 is rejected fail-closed WITH
// THE VLMAX REASON; the m1 anchor (VLMAX 16) verifies clean and round-trips.
// NARROWANCHOR: integer_core_lmul "mf2" cannot host the 16-entry codebook gather at VLEN 128
// WIDEANCHOR: integer_core_lmul = "m1"
