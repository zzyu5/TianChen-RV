// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-18 F5b — the COMPLETE ggml ggml_vec_soft_max_f32 forward-pass op
// (y[i] = e^{x[i]-max}; RETURNS the f64 sum Sum_i e^{x[i]-max}) as STRUCTURED
// emitc IR (I5; ZERO raw() strings). Faithful to ggml's BARE function
// (vec.cpp:531): `max` is an INPUT (the caller computes it with the scalar
// ggml_vec_max_f32 and passes it; ops.cpp:5400), and the function does NOT
// normalize (the caller applies 1/sum via ggml_vec_scale_f32; ops.cpp:5415-16).
//
// The single typed op tcrv_rvv.ggml_vec_soft_max_f32 lowers to ONE m2 f32 strip
// loop whose body is ggml's EXACT __riscv_v path (vec.cpp:584-592):
//   vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
//   for (avl over n) {
//     val = ggml_v_expf_m2(vfsub_vf(vle32(x), max));   // shared exp polynomial
//     vse32(y, val);
//     vsum = vfwredusum_vs_f32m2_f64m1(val, vsum);      // f32->f64 widening reduce
//   }
//   return (double)vfmv_f_s_f64m1(vsum);
// The sum is accumulated in ggml_float = DOUBLE via the WIDENING reduce
// vfwredusum_vs_f32m2_f64m1 into a single f64m1 accumulator carried across
// strips (NOT F3's scalar-ascending fold) -- the byte-exactness crux for the
// returned sum. exp(x-max) reuses the SAME node-for-node ggml_v_expf_m2 chain
// F5 silu emits, so y[] and each val are bit-identical to ggml. -inf masked
// entries (attention masks) flow through the polynomial's underflow path
// (exp -> 0) with no special-casing. Byte-exactness pinned by the ssh-rvv
// artifact under .trellis/tasks/.../artifacts/inc18-forward-pass-f5b/.

module {
  tcrv.exec.kernel @ggml_vec_soft_max_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_soft_max_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %max = tcrv_rvv.runtime_abi_value {c_name = "max", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_soft_max_f32, sew = 32 : i64, source_kernel = "ggml_vec_soft_max_f32_kernel", status = "selected-lowering-boundary"} {
        %sm = tcrv_rvv.ggml_vec_soft_max_f32 %y, %x, %max, %n, %vl {kind = "ggml_vec_soft_max_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The function RETURNS double (faithful to ggml's ggml_float return).
// CHECK: emitc.func @tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(
// CHECK-SAME: -> !emitc.opaque<"double">
// The loop-carried f64m1 sum accumulator: vsum = vfmv_v_f_f64m1(0.0, 1).
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
// The slow-path overflow/underflow vmerge fixup, emitted UNCONDITIONALLY.
// CHECK: call_opaque "__riscv_vmfle_vf_f32m2_b16"
// CHECK: literal "0x82000000"
// CHECK: call_opaque "__riscv_vmerge_vxm_u32m2"
// CHECK: literal "0x7f000000"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// CHECK: literal "192.0f"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m2"
// soft_max epilogue: store y[i] = exp(x[i]-max), then the f64 widening reduce.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vfwredusum_vs_f32m2_f64m1"
// CHECK: assign
// return (double)vfmv_f_s_f64m1(vsum).
// CHECK: call_opaque "__riscv_vfmv_f_s_f64m1_f64"
// CHECK: return %{{.*}} : !emitc.opaque<"double">
// Every value is a structured emitc node, NOT a raw C blob. The provenance
// verbatims are comment lines.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
