// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_elementwise_loop_body"/kind = "plain_loop"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/reduce_map_model = "rotate"/reduce_map_model = "spin"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/kind = "elementwise_rope_rotate_core"/kind = "elementwise_bogus_rotate_core"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADBRICK

// M-FLAT forward-elementwise scaffold ROTATE model (line C, G1-tail) — the
// CONSTRUCTED f32 forward-pass rope (ggml's ggml_compute_forward_rope_f32 NORMAL
// variant for ONE head row: the position-dependent 2x2 rotation on CONSECUTIVE
// pairs x[2p]/x[2p+1] + the scalar-libm cosf/sinf angle cache + the iterative f32
// theta recurrence; the LLM_ARCH_LLAMA rope), the FIRST (and only) forward ROTATE
// operator flipped dispatch-wired -> constructed (C_construct 32->33). rope does
// NOT fit the MAP model (a MAP is a vectorized `i += vlmax` strip with NO
// loop-carried state; rope is a SCALAR per-PAIR loop with a loop-carried f32
// recurrence), so it lands its OWN increment: a THIRD reduce_map_model "rotate"
// on the SAME typed elementwise strip-loop op tcrv_rvv.typed_elementwise_loop_body,
// which REUSES the loop-carried-scalar region SHAPE the "reduce" model pioneered
// (a SECOND region arg = the loop-carried value + the yield that carries it back,
// realized as an emitc.variable lvalue since emitc.for has no iter_args), EXCEPT
// the carried value is a DATA-INDEPENDENT f32 recurrence (theta *= theta_scale),
// NOT a reduction of the buffer. The NEW per-op machinery is the rotate core brick
// tcrv_rvv.elementwise_rope_rotate_core (it carries the whole rope ABI + the
// per-pair rotation; anti-bypass: its pair_index is region arg 0 and its theta is
// region arg 1). The monolith tcrv_rvv.ggml_rope_norm_f32 op + emitGgmlRopeNormF32
// opaque helper + recognizer + verifier were RETIRED.
//
// This is BYTE-EXACT to the retired monolith emit modulo ONLY the source-op
// provenance token (tcrv_rvv.ggml_rope_norm_f32 ->
// tcrv_rvv.elementwise_rope_rotate_core). BYTE-EXACTNESS has TWO honest axes:
// (1) cosf/sinf are SCALAR libm (one call_opaque each — the sanctioned opaque
// seam, a DIFFERENT byte-exactness axis from silu/soft_max's replicated vectorized
// exp polynomial: libm-LINKED, bit-exact under same-libm, libm-tolerance on the
// angles otherwise); (2) each output's a*b-c*d rotation is GROUPED into ONE
// emitc.expression so mlir-translate renders ONE C statement token-identical to
// ggml's single C expression -> clang contracts identically under every
// -ffp-contract mode -> byte-exact regardless of the build flag. The emit is a
// decomposed pattern-library rotate primitive with NO opaque hand helper ([L-8]
// constructed-strong). Numerical bit-exact-vs-ggml is pending-hardware.

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
        tcrv_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "rotate", element_sew = 32 : i64} {
        ^bb0(%pair_index: index, %theta: f32):
          // The per-pair rope rotate core brick: read the carried theta, cosf/sinf
          // the angle, load the consecutive pair x[2p]/x[2p+1], write the 2x2
          // rotation y[2p]=x0*cos-x1*sin / y[2p+1]=x0*sin+x1*cos, then step the
          // recurrence theta_next = theta * theta_scale. Its pair_index is the loop
          // induction variable (region arg 0) and its theta is the loop-carried f32
          // recurrence (region arg 1), the anti-bypass ties. NO strip_lmul knob
          // (cos/sin are scalar libm, the loop is scalar per-pair).
          %theta_next = tcrv_rvv.elementwise_rope_rotate_core %x, %y, %tb, %ts, %n pair %pair_index theta %theta {kind = "elementwise_rope_rotate_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, f32 -> f32
          tcrv_rvv.typed_elementwise_loop_yield %theta_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The rope TU calls scalar libm (cosf/sinf), so the emitted module MUST
// self-include <math.h> to be a self-contained standalone TU (byte-exact to the
// retired monolith's self-include behavior; keyed on the rotate core brick).
// CHECK: emitc.include <"math.h">
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
// Every value is a structured emitc node (variable/load/for/call_opaque); the
// rotation/recurrence are emitc.mul/sub/add/assign nodes, NOT a raw C blob. The
// provenance verbatims carry the constructed rotate brick's op identity, NOT the
// retired monolith op, and NO opaque C blob leaks.
// CHECK-NOT: tcrv_rvv.ggml_rope_norm_f32
// CHECK-NOT: emitc.verbatim {{.*}}__riscv

// The bounded surface is fail-closed on the loop kind, the reduce_map_model fact,
// and the rotate core brick kind (I7), enforced by the loop-body + brick verifiers.
// BADKIND: currently supports only kind "typed_elementwise_loop_body"
// BADMODEL: currently supports only reduce_map_model "map"
// BADBRICK: currently supports only kind "elementwise_rope_rotate_core"
