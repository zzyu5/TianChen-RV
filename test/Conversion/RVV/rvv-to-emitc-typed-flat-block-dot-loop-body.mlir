// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_flat_block_dot_loop_body"/kind = "plain_block_dot_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "sumi_times_scales"/fold_model = "unsupported_fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/integer_core_lmul = "m2"/integer_core_lmul = "m8"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADLMUL

// M-FLAT loop-scaffold step 1/6 -- the FIRST typed op that puts a loop INTO
// the CORE typed body. weft_rvv.typed_flat_block_dot_loop_body carries the
// flat block-dot's `nb = n / QK` block loop as ONE region-carrying op whose
// entry arguments are the block_index induction variable and the SSA
// loop-carried f32 accumulator. The isolated hard bone this step proves is
// that the SSA loop-carried acc lowers BYTE-EXACT to the monolithic
// emitFlatBlockDot's mutable `float sumf` form: emitc.for has no iter_args, so
// the carried-IN `acc` block argument maps to a LOAD of the sumf emitc.variable
// lvalue at the top of the loop body and the typed loop-yield's carried-OUT
// `acc_next` maps to an emitc.assign back into it at the bottom. The seed is
// the literal `0.0f` emitted directly (ggml's `float sumf = 0.0f;` is a
// hardcoded zero; the op carries no init operand, mirroring the block-dot ops).
//
// This is an emit-consistency ("CORE == emission-plans") lit that locks the
// loop + accumulator SKELETON byte-exact to emitFlatBlockDot: the sumf
// emitc.variable + `0.0f` seed (:5410-5417), nb = n/QK (:5419-5422), the
// emitc.for over nb (:5879-5883), the load-at-top / assign-at-bottom of the
// sumf lvalue (:5757, :5829), and the `*s = sumf` scalar store (:5931-5945).
// The minimal region body is a single weft_rvv.cross_block_f32_accumulate over
// a STUB term (a runtime f32), so the FOLD EXPRESSION itself is NOT locked
// here -- full-body single-instance byte-exactness is a later step. Numerical
// bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here.

module {
  weft.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = weft_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_flat_block_dot_loop_body, sew = 8 : i64, source_kernel = "rvv_typed_flat_block_dot_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", multi_block_factor = 1 : i64, strip_elision = "elided", fold_structure = "per-block", numerics_tier = "strict"} {
        ^bb0(%block_index: index, %acc: f32):
          // Minimal body: brick 3 folds the block-carried acc with a STUB f32 term.
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_typed_flat_block_dot_loop_body_kernel_rvv_typed_flat_block_dot_loop_body(

// The SSA loop-carried acc lowers to the mutable `float sumf = 0.0f;` emitc
// variable -- byte-exact to emitFlatBlockDot's sumf accumulator decl.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: %[[ZERO:.*]] = literal "0.0f" : !emitc.opaque<"float">
// CHECK: assign %[[ZERO]] : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>

// size_t nb = n / QK;  -- n is the scope AVL, QK the block element count.
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// The outer block loop for (ib = 0; ib < nb; ib += 1) -- byte-exact to the
// emitFlatBlockDot mbf==1 form.
// CHECK: for %{{.*}} = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// The carried-IN acc block arg -> a LOAD of the sumf lvalue at the TOP.
// CHECK: %[[ACCIN:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// brick 3 folds acc + STUB term (the fold expression is NOT locked in step 1).
// CHECK: %[[NEXT:.*]] = add %[[ACCIN]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// The carried-OUT acc_next -> an emitc.assign back into the sumf lvalue at the
// BOTTOM (the SSA-acc <-> emitc-variable mapping, no hacky bypass).
// CHECK: assign %[[NEXT]] : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>

// *s = sumf;  -- the structured scalar store through the output pointer.
// CHECK: %[[OUTIDX:.*]] = literal "0" : index
// CHECK: %[[SUB:.*]] = subscript %{{.*}}[%[[OUTIDX]]] : (!emitc.ptr<!emitc.opaque<"float">>, index) -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: %[[FINAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: assign %[[FINAL]] : !emitc.opaque<"float"> to %[[SUB]] : <!emitc.opaque<"float">>
// CHECK: return

// The bounded surface is fail-closed on the loop kind, the fold_model fact, and
// the scheduling knobs (I7). The loop-carried f32 accumulator dtype is enforced
// by the verifier (region arg + yield f32) and exercised by the positive path.
// BADKIND: currently supports only kind "typed_flat_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "sumi_times_scales"
// BADLMUL: only accepts integer_core_lmul "m1", "m2", or "mf4"
