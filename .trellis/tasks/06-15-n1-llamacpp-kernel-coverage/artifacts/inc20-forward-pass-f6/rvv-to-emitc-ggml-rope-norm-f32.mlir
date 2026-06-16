// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-20 F6 — the COMPLETE ggml ggml_compute_forward_rope_f32 op (the NORMAL
// variant, the rope llama-2 / LLM_ARCH_LLAMA uses; llama-model.cpp:2387-2423
// maps LLM_ARCH_LLAMA -> LLAMA_ROPE_TYPE_NORM, "a normal RoPE operating on pairs
// of CONSECUTIVE head values") for ONE head row, as STRUCTURED emitc IR (I5;
// ZERO raw() strings). The single typed op tcrv_rvv.ggml_rope_norm_f32 lowers to
// a SINGLE scalar per-pair loop:
//   (1) a loop-carried f32 angle recurrence `theta` (an emitc.variable lvalue +
//       emitc.assign, since emitc.for has no iter_args), seeded from the runtime
//       theta_base (= pos as f32) and stepped `theta *= theta_scale` per pair
//       (the ITERATIVE f32 recurrence, NOT a closed-form pow, NOT double;
//       ops.cpp:5711/5719). theta_scale = powf(freq_base, -2/n_dims) is a
//       PRECOMPUTED runtime input, so the kernel makes no powf call.
//   (2) the SCALAR libm angle cache: cosf(theta) / sinf(theta) (one
//       emitc.call_opaque each -- the sanctioned opaque seam, the libm-linked
//       byte-exactness axis; ops.cpp:5703-5704).
//   (3) the per-pair f32 rotation on the CONSECUTIVE pair x[2p], x[2p+1]:
//       y[2p]   = x0*cos - x1*sin  and  y[2p+1] = x0*sin + x1*cos, with each
//       output's a*b-c*d GROUPED into ONE emitc.expression (the F3 rms_norm FMA-
//       fix discipline), so mlir-translate renders ONE C statement token-
//       identical to ggml's single C expression -> clang contracts identically
//       under every -ffp-contract mode -> byte-exact regardless of the build flag
//       (NOT just off).
//
// This is the F6 COMPOSITION rung — the last forward-pass primitive. The angle
// transcendental is SCALAR libm (a DIFFERENT byte-exactness axis from F5's
// vectorized exp polynomial: libm-linked, not replicated). Every emitted value
// is a NODE in the IR graph — no emitc.verbatim with C control flow, no raw
// string blob. Byte-exactness vs ggml's real rope (same-libm, -ffp-contract=off)
// is pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc20-forward-pass-f6/.

module {
  tcrv.exec.kernel @ggml_rope_norm_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_rope_norm_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n_dims", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %tb = tcrv_rvv.runtime_abi_value {c_name = "theta_base", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %ts = tcrv_rvv.runtime_abi_value {c_name = "theta_scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_rope_norm_f32, sew = 32 : i64, source_kernel = "ggml_rope_norm_f32_kernel", status = "selected-lowering-boundary"} {
        %rot = tcrv_rvv.ggml_rope_norm_f32 %x, %y, %tb, %ts, %n, %vl {kind = "ggml_rope_norm_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(
// The loop-carried f32 angle recurrence theta as an emitc.variable lvalue
// (emitc.for has no iter_args), seeded from theta_base.
// CHECK: %[[THETA:.*]] = "emitc.variable"{{.*}}lvalue<!emitc.opaque<"float">>
// n_pairs = n_dims / 2 and the SCALAR per-pair loop.
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">)
// CHECK: for %[[P:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// The scalar libm angle cache cosf / sinf (the sanctioned opaque seam).
// CHECK: call_opaque "cosf"
// CHECK: call_opaque "sinf"
// The consecutive pair element pointer x + 2*p.
// CHECK: mul %[[P]], %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">)
// The rotation y[2p] = x0*cos - x1*sin GROUPED into ONE emitc.expression (so it
// renders as one C statement token-identical to ggml's -> identical contraction
// under every -ffp-contract mode). The two muls and the sub live INSIDE the
// expression, yielded as one value.
// CHECK: %[[LOEXPR:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: mul %{{.*}}, %{{.*}} -> !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} -> !emitc.opaque<"float">
// CHECK: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield
// CHECK: assign %[[LOEXPR]] : !emitc.opaque<"float"> to %{{.*}}
// The rotation y[2p+1] = x0*sin + x1*cos GROUPED into a SECOND emitc.expression.
// CHECK: %[[HIEXPR:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield
// CHECK: assign %[[HIEXPR]] : !emitc.opaque<"float"> to %{{.*}}
// The iterative recurrence step theta = theta * theta_scale, assigned back.
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[THETA]]
// The loop is a structured emitc.for; cosf/sinf are emitc.call_opaque; the
// rotation/recurrence are emitc.mul/sub/add/assign nodes, not a raw C blob. The
// provenance verbatims are comment lines only.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
