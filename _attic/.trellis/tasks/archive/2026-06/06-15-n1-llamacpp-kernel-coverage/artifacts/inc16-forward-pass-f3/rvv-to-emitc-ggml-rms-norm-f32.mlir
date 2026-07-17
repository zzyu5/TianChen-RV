// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-16 F3 — the COMPLETE ggml ggml_compute_forward_rms_norm_f32 (non-fused, no
// weight) forward-pass op as STRUCTURED emitc IR (I5; ZERO raw() strings). The
// single typed op tcrv_rvv.ggml_rms_norm_f32 lowers to THREE structured pieces:
//   (1) a SCALAR double accumulator + a scalar ascending fold loop computing
//       Sum_i x[i]^2 in `ggml_float`(= double): each product x[i]*x[i] rounds in
//       f32 FIRST, is WIDENED to double, then accumulated in double (the cast
//       chain is load-bearing for byte-exactness; ops.cpp:3791-3795). This is
//       NOT a vectorized vfredusum (which would fold in f32 + a tree order).
//   (2) the scalar mean = (float)(sum / (double)ne00) (divide in double, cast to
//       f32 after) and scale = 1.0f / sqrtf(mean + eps) (the add, sqrtf, and
//       reciprocal all in f32; ops.cpp:3797-3798).
//   (3) the VECTORIZED normalize strip y[i] = x[i]*scale: a vsetvl_e32m8(ne00-i)
//       strip over the runtime element count with vle32 / vfmul_vf (scalar
//       broadcast) / vse32 (the F1 machinery, two-buffer x in / y out).
//
// This is the FIRST forward-pass REDUCTION op — a NEW shape vs the elementwise
// scale (F1): a loop-carried scalar-double accumulator (an emitc.variable lvalue
// + emitc.assign, since emitc.for has no iter_args) plus a scalar rsqrt before
// the elementwise normalize. Every emitted value is a NODE in the IR graph — no
// emitc.verbatim with C control flow, no raw string blob. Byte-exactness vs
// ggml's real rms_norm is pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc16-forward-pass-f3/.

module {
  tcrv.exec.kernel @ggml_rms_norm_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_rms_norm_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "ne00", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %eps = tcrv_rvv.runtime_abi_value {c_name = "eps", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_rms_norm_f32, sew = 32 : i64, source_kernel = "ggml_rms_norm_f32_kernel", status = "selected-lowering-boundary"} {
        %norm = tcrv_rvv.ggml_rms_norm_f32 %x, %y, %eps, %n, %vl {kind = "ggml_rms_norm_f32", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(
// The SCALAR double accumulator (ggml_float sum = 0.0) as an emitc.variable
// lvalue (emitc.for has no iter_args, so the loop-carried accumulator is a
// variable + assign, NOT loop-carried SSA).
// CHECK: %[[SUM:.*]] = "emitc.variable"{{.*}}lvalue<!emitc.opaque<"double">>
// The SCALAR ascending fold loop over the runtime element count (step is the
// literal 1, NOT a vlmax -- this loop is NOT vectorized).
// CHECK: literal "1" : !emitc.opaque<"size_t">
// CHECK: for %[[RI:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// The f32 product x[i]*x[i] (one f32 round) widened to double then accumulated
// in double -- the cast SITS BETWEEN the f32 mul and the f64 add (FMA barrier).
// CHECK: %[[PROD:.*]] = mul %{{.*}}, %{{.*}}{{.*}} -> !emitc.opaque<"float">
// CHECK: cast %[[PROD]] : !emitc.opaque<"float"> to !emitc.opaque<"double">
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// CHECK: assign %{{.*}} : !emitc.opaque<"double"> to %[[SUM]]
// mean = (float)(sum / (double)ne00): divide in double, cast to f32 AFTER.
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// CHECK: cast %{{.*}} : !emitc.opaque<"double"> to !emitc.opaque<"float">
// scale = 1.0f / sqrtf(mean + eps): f32 add, scalar libm sqrtf, f32 reciprocal.
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: call_opaque "sqrtf"
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The VECTORIZED normalize strip y[i] = x[i]*scale.
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: for %[[NI:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// The reduction loop, the scalar rsqrt, and the normalize strip are all
// structured emitc nodes (variable/load/mul/cast/add/div/for/call_opaque), not a
// raw C blob. The provenance verbatims are comment lines only.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
