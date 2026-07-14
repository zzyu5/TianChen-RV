// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_elementwise_loop_body"/kind = "plain_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/reduce_map_model = "reduce"/reduce_map_model = "fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/kind = "elementwise_soft_max_reduce_core"/kind = "elementwise_bogus_reduce_core"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADBRICK

// M-FLAT forward-elementwise scaffold REDUCE model (line C, G1-tail) — the
// CONSTRUCTED f32 forward-pass soft_max (ggml's BARE ggml_vec_soft_max_f32:
// y[i] = e^{x[i]-max}, RETURNING the f64 sum Sum_i e^{x[i]-max}; vec.cpp:531 + the
// __riscv_v path vec.cpp:584-592), the SECOND forward REDUCE operator flipped
// dispatch-wired -> constructed (C_construct 31->32). It REUSES the reduce model
// rms_norm built on the SAME typed elementwise strip-loop SCAFFOLD: the loop op
// weft_rvv.typed_elementwise_loop_body under reduce_map_model "reduce" carries a
// SECOND region argument (the loop-carried accumulator) and the yield names ONE
// operand (the updated accumulator), EXCEPT the accumulator is the f64m1 WIDENING
// vector (ggml's vfloat64m1_t vsum, the vfwredusum_vs_f32m2_f64m1 destination),
// NOT rms_norm's scalar double. The NEW per-op machinery is the exp-sum-reduce
// core brick weft_rvv.elementwise_soft_max_reduce_core (it carries the whole
// soft_max ABI + the fused exp-store-widening-reduce strip; anti-bypass: its
// strip_index is region arg 0 and its acc is region arg 1). The monolith
// weft_rvv.ggml_vec_soft_max_f32 op + emitGgmlVecSoftMaxF32 opaque helper +
// recognizer + verifier were RETIRED.
//
// This is BYTE-EXACT to the retired monolith emit modulo ONLY the source-op
// provenance token (weft_rvv.ggml_vec_soft_max_f32 ->
// weft_rvv.elementwise_soft_max_reduce_core). exp(x-max) reuses the SAME
// node-for-node ggml_v_expf_m2 chain silu emits (the shared emitGgmlVExpfM2), and
// the f64 sum is folded via the WIDENING reduce vfwredusum_vs_f32m2_f64m1 into a
// single f64m1 accumulator carried across strips — matching THAT exact fold is the
// byte-exactness crux for the returned sum. -inf masked entries flow through the
// polynomial's underflow path (exp -> 0) with no special-casing. The emit is a
// decomposed pattern-library reduce primitive with NO opaque hand helper ([L-8]
// constructed-strong). Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_vec_soft_max_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_soft_max_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %max = weft_rvv.runtime_abi_value {c_name = "max", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_soft_max_f32, sew = 32 : i64, source_kernel = "ggml_vec_soft_max_f32_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%strip_index: index, %acc: !weft_rvv.vector<f64, "m1">):
          // The per-strip soft_max exp-sum-reduce core brick: subtract max, run the
          // shared exp polynomial, store y[i] = e^{x[i]-max}, then fold val into
          // the f64m1 widening accumulator acc_next = vfwredusum(val, acc). Its
          // strip_index is the loop induction variable (region arg 0) and its acc
          // is the loop-carried f64m1 accumulator (region arg 1), the anti-bypass
          // ties. NO strip_lmul knob (m2/f64m1-pinned).
          %acc_next = weft_rvv.elementwise_soft_max_reduce_core %y, %x, %max, %n strip %strip_index acc %acc {kind = "elementwise_soft_max_reduce_core"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vector<f64, "m1"> -> !weft_rvv.vector<f64, "m1">
          weft_rvv.typed_elementwise_loop_yield %acc_next : !weft_rvv.vector<f64, "m1">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The function RETURNS double (faithful to ggml's ggml_float return) — the reduce
// model's f64 sum is the ONLY forward-pass op whose function returns a scalar.
// CHECK: emitc.func @weft_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(
// CHECK-SAME: -> !emitc.opaque<"double">
// The loop-carried f64m1 sum accumulator (the reduce model's vector acc region arg
// realized as an emitc.variable lvalue): vsum = vfmv_v_f_f64m1(0.0, 1).
// CHECK: emitc.variable
// CHECK: literal "0.0"
// CHECK: literal "1"
// CHECK: call_opaque "__riscv_vfmv_v_f_f64m1"
// CHECK: assign
// The pre-loop VLMAX vsetvl and the m2 f32 strip loop.
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: sub %arg0, %[[I]]
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
// CHECK: call_opaque "__riscv_vle32_v_f32m2"
// x[i] - max before the exp.
// CHECK: call_opaque "__riscv_vfsub_vf_f32m2"
// ggml_v_expf_m2 node-for-node (SHARED with silu): the 0x1.8p23f round trick.
// CHECK: literal "0x1.8p23f"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: literal "0x1.715476p+0f"
// CHECK: call_opaque "__riscv_vfmacc_vf_f32m2"
// CHECK: call_opaque "__riscv_vfsub_vv_f32m2"
// The two-term Cayley range reduction (vfnmsac x2).
// CHECK: literal "0x1.62e4p-1f"
// CHECK: call_opaque "__riscv_vfnmsac_vf_f32m2"
// CHECK: literal "0x1.7f7d1cp-20f"
// CHECK: call_opaque "__riscv_vfnmsac_vf_f32m2"
// The integer exponent vsll<<23 + 0x3f800000 reinterpret.
// CHECK: call_opaque "__riscv_vreinterpret_v_f32m2_u32m2"
// CHECK: literal "23"
// CHECK: call_opaque "__riscv_vsll_vx_u32m2"
// CHECK: literal "0x3f800000"
// CHECK: call_opaque "__riscv_vadd_vx_u32m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// The |n|>126 overflow mask + the degree-5 Estrin polynomial.
// CHECK: literal "126.0f"
// CHECK: call_opaque "__riscv_vmfgt_vf_f32m2_b16"
// CHECK: literal "0x1.ffffecp-1f"
// CHECK: literal "0x1.fffdb6p-2f"
// CHECK: literal "0x1.555e66p-3f"
// CHECK: literal "0x1.573e2ep-5f"
// CHECK: literal "0x1.0e4020p-7f"
// RESTORED ggml vcpop short-circuit (vec.h:1348 `if (!vcpop(c)) return fast`): the
// fast result (k + j*k) is seeded into an emitc.variable, and the ~14-op slow-path
// overflow/underflow fixup is emitted INSIDE a data-dependent emitc.if guarded by
// __riscv_vcpop_m_b16(c) != 0 (SKIPPED on the all-fast strip = every soft_max decode
// input). BYTE-EXACT: vcpop==0 keeps exactly the seeded fast value, which the retired
// unconditional slow path already produced for c-false/|n|<=192 lanes; the f64 reduce
// consumes the SAME per-lane exp value either way.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: emitc.variable
// CHECK: assign
// CHECK: call_opaque "__riscv_vcpop_m_b16"
// CHECK: literal "0"
// CHECK: cmp ne
// CHECK: if %
// CHECK: call_opaque "__riscv_vmfle_vf_f32m2_b16"
// CHECK: literal "0x82000000"
// CHECK: call_opaque "__riscv_vmerge_vxm_u32m2"
// CHECK: literal "0x7f000000"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// CHECK: literal "192.0f"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// CHECK: assign
// CHECK: load
// soft_max epilogue: store y[i] = exp(x[i]-max), then the f64 WIDENING reduce into
// the loop-carried f64m1 accumulator (the byte-exactness crux for the sum).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vfwredusum_vs_f32m2_f64m1"
// CHECK: assign
// return (double)vfmv_f_s_f64m1(vsum).
// CHECK: call_opaque "__riscv_vfmv_f_s_f64m1_f64"
// CHECK: return %{{.*}} : !emitc.opaque<"double">
// Every value is a structured emitc node (variable/load/for/call_opaque), NOT a raw
// C blob. The provenance verbatims carry the constructed reduce brick's op identity,
// NOT the retired monolith op, and NO opaque C blob leaks.
// CHECK-NOT: weft_rvv.ggml_vec_soft_max_f32
// CHECK-NOT: emitc.verbatim {{.*}}__riscv{{.*}};

// The bounded surface is fail-closed on the loop kind, the reduce_map_model fact,
// and the reduce core brick kind (I7), enforced by the loop-body + brick verifiers.
// BADKIND: currently supports only kind "typed_elementwise_loop_body"
// BADMODEL: currently supports only reduce_map_model "map"
// BADBRICK: currently supports only kind "elementwise_soft_max_reduce_core"
