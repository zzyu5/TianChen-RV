// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_elementwise_loop_body"/kind = "plain_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/reduce_map_model = "map"/reduce_map_model = "fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/element_sew = 32 : i64/element_sew = 16 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSEW

// M-FLAT forward-elementwise scaffold (line C, ① 之后) — the CONSTRUCTED f32
// forward-pass silu (y[i] = x[i]*sigmoid(x[i]), sigmoid(x) = 1/(1+e^{-x})), the
// SECOND forward-elementwise operator flipped dispatch-wired -> constructed
// (C_construct 29->30). This is a C2 marginal-cost payoff: it REUSES the typed
// elementwise strip-loop SCAFFOLD scale landed — the SAME loop op
// weft_rvv.typed_elementwise_loop_body (reduce_map_model "map"), the SAME yield
// terminator, the SAME emitTypedElementwiseLoopBody outer-loop machinery + [L-8]
// validator, and the SAME shared node-for-node ggml_v_expf_m2 exp polynomial
// (which soft_max also consumes) — adding ONLY the per-strip
// weft_rvv.elementwise_silu_map map core brick (its m2 exp decode). The monolith
// weft_rvv.ggml_vec_silu_f32 op + emitGgmlVecSiluF32 opaque helper + recognizer +
// verifier were RETIRED. The brick's strip_index MUST be the loop induction
// variable (region arg 0, anti-bypass), so the emit provably addresses x + i /
// y + i, not the loop-invariant strip 0.
//
// This is BYTE-EXACT to the retired monolith emit modulo ONLY the source-op
// provenance token (weft_rvv.ggml_vec_silu_f32 -> weft_rvv.elementwise_silu_map):
// ggml's `if (!vcpop_m(c))` is a pure perf short-circuit whose fast/slow paths are
// bitwise-equal, so the slow-path vmerge value graph is emitted UNCONDITIONALLY.
// Pinned at m2 (ggml's vsetvl_e32m2 path + the m2-tied vbool16_t/vuint32m2_t
// mask/reinterpret types), so there is no strip_lmul knob. The emit is a
// decomposed pattern-library map primitive with NO opaque hand helper ([L-8]
// constructed-strong). Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_vec_silu_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_silu_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_silu_f32, sew = 32 : i64, source_kernel = "ggml_vec_silu_f32_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "map", element_sew = 32 : i64} {
        ^bb0(%strip_index: index):
          // The per-strip silu map core brick: y[i..i+vl] = silu(x[i..i+vl]). Its
          // strip_index is the loop induction variable (region arg 0), the
          // anti-bypass tie. NO strip_lmul knob — silu is m2-pinned (the exp
          // polynomial mask/reinterpret types are m2-tied).
          weft_rvv.elementwise_silu_map %x, %y, %n strip %strip_index : index {kind = "elementwise_silu_map"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_elementwise_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32(
// The outer setvl config (scope frame) is unchanged: vsetvl_e32m1.
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// The pre-loop VLMAX vsetvl (the m2 silu strip anchor) and the m2 f32 strip loop.
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Remaining-AVL re-strip vsetvl inside the loop.
// CHECK: sub %arg0, %[[I]]
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
// In-place element pointer x + i (anti-bypass: addresses x + i, not strip 0).
// CHECK: add %arg1, %[[I]]
// CHECK: call_opaque "__riscv_vle32_v_f32m2"
// silu = neg -> ggml_v_expf_m2 -> +1 -> div.
// CHECK: call_opaque "__riscv_vfneg_v_f32m2"
// ggml_v_expf_m2 node-for-node: the 0x1.8p23f round trick.
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
// CHECK: call_opaque "__riscv_vfmul_vv_f32m2"
// CHECK: literal "0x1.ffffecp-1f"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: literal "0x1.fffdb6p-2f"
// CHECK: literal "0x1.555e66p-3f"
// CHECK: literal "0x1.573e2ep-5f"
// CHECK: literal "0x1.0e4020p-7f"
// The slow-path overflow/underflow vmerge fixup, emitted UNCONDITIONALLY.
// CHECK: call_opaque "__riscv_vmfle_vf_f32m2_b16"
// CHECK: call_opaque "__riscv_vmv_v_x_u32m2"
// CHECK: literal "0x82000000"
// CHECK: call_opaque "__riscv_vmerge_vxm_u32m2"
// CHECK: literal "0x7f000000"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// CHECK: literal "192.0f"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// silu epilogue: 1 + exp(-x) then x / (1 + exp(-x)).
// CHECK: literal "1.0f"
// CHECK: call_opaque "__riscv_vfadd_vf_f32m2"
// CHECK: call_opaque "__riscv_vfdiv_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// The provenance verbatims carry the constructed map brick's op identity, NOT the
// retired monolith op, and NO opaque C blob leaks into the body.
// CHECK-NOT: weft_rvv.ggml_vec_silu_f32
// CHECK-NOT: emitc.verbatim {{.*}}__riscv

// The bounded surface is fail-closed on the loop kind, the reduce_map_model fact,
// and the element_sew fact (I7), enforced by the loop-body verifier.
// BADKIND: currently supports only kind "typed_elementwise_loop_body"
// BADMODEL: currently supports only reduce_map_model "map"
// BADSEW: currently supports only element_sew 32
