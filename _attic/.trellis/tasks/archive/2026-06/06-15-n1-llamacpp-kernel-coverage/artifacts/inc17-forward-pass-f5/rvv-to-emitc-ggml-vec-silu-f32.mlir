// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-17 F5 — the COMPLETE ggml ggml_vec_silu_f32 forward-pass op
// (y[i] = x[i]*sigmoid(x[i]), sigmoid(x) = 1/(1+e^{-x})) as STRUCTURED emitc IR
// (I5; ZERO raw() strings). The single typed op tcrv_rvv.ggml_vec_silu_f32
// lowers to ONE m2 f32 strip loop whose body is ggml's EXACT vectorized silu:
//   neg_x = vfneg(x);  exp_neg_x = ggml_v_expf_m2(neg_x);
//   y = vfdiv(x, vfadd(exp_neg_x, 1.0f));
// where ggml_v_expf_m2 is a fully vectorized minimax exp polynomial replicated
// node-for-node from vec.h:1324-1360 — a 0x1.8p23f round trick, a two-term
// Cayley range reduction (vfnmsac x2), a degree-5 Estrin polynomial (vfmacc),
// an integer exponent vsll<<23 + 0x3f800000 reinterpret, and the
// overflow/underflow vmerge fixup. It makes NO libm expf call — every step is
// an emitc.call_opaque node, so the result is BIT-IDENTICAL to ggml's
// vectorized silu (the deployment oracle). ggml's `if (!vcpop_m(c))` is a pure
// performance short-circuit (fast k+j*k == slow c-false lane k+k*j bit-for-bit),
// so the slow-path vmerge value graph is emitted UNCONDITIONALLY (straight-line,
// no data-dependent branch). Pinned at m2 (ggml's vsetvl_e32m2 path + the
// m2-tied vbool16_t/vuint32m2_t types). Byte-exactness pinned by the ssh-rvv
// artifact under .trellis/tasks/.../artifacts/inc17-forward-pass-f5/.

module {
  tcrv.exec.kernel @ggml_vec_silu_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_silu_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_silu_f32, sew = 32 : i64, source_kernel = "ggml_vec_silu_f32_kernel", status = "selected-lowering-boundary"} {
        %silu = tcrv_rvv.ggml_vec_silu_f32 %x, %y, %n, %vl {kind = "ggml_vec_silu_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32(
// The pre-loop VLMAX vsetvl and the m2 f32 strip loop.
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: sub %arg0, %[[I]]
// CHECK: call_opaque "__riscv_vsetvl_e32m2"
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
// Every value is a structured emitc node (for / call_opaque / literal / sub /
// add / cast), NOT a raw C blob. The provenance verbatims are comment lines.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
