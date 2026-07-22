// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_elementwise_loop_body"/kind = "plain_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/reduce_map_model = "reduce"/reduce_map_model = "fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/element_sew = 32 : i64/element_sew = 16 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSEW

// M-FLAT forward-elementwise scaffold (line C, G1-tail) — the CONSTRUCTED f32
// forward-pass rms_norm (mean = (Sum_i x[i]^2)/ne00, scale = 1/sqrtf(mean+eps),
// y[i] = x[i]*scale), the FIRST forward REDUCE operator flipped dispatch-wired ->
// constructed (C_construct 30->31). It BUILDS the reduce model of the SAME typed
// elementwise strip-loop SCAFFOLD scale/silu landed: the loop op
// weft_rvv.typed_elementwise_loop_body now under reduce_map_model "reduce" carries
// a SECOND region argument — the loop-carried f64 accumulator (region arg 1) —
// and the yield names ONE operand (the updated accumulator), the genuine
// loop-carried-acc modeling the block-dot loop scaffold uses for sumf. The NEW
// per-op machinery is the reduce core brick weft_rvv.elementwise_rms_norm_reduce_core
// (it carries the whole rms_norm ABI + the Σx² fold; anti-bypass: its strip_index
// is region arg 0 and its acc is region arg 1). The monolith weft_rvv.ggml_rms_norm_f32
// op + emitGgmlRmsNormF32 opaque helper + recognizer + verifier were RETIRED.
//
// This is BYTE-EXACT to the retired monolith emit modulo ONLY the source-op
// provenance token (weft_rvv.ggml_rms_norm_f32 ->
// weft_rvv.elementwise_rms_norm_reduce_core). The reduce is the byte-exact
// scalar-double ascending fold (each x[i]*x[i] rounds in f32, widens to double —
// the FMA barrier — then adds in double); a vectorized vfredusum would fold in f32
// + a tree order and break byte-exactness, so ONLY the normalize strip is
// vectorized. The reduce model is reusable: softmax's Σe^x is the same loop-carried
// accumulator shape with a vfwredusum reduce brick (a later step). The emit is a
// decomposed pattern-library reduce primitive with NO opaque hand helper ([L-8]
// constructed-strong). Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_rms_norm_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_rms_norm_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "ne00", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %eps = weft_rvv.runtime_abi_value {c_name = "eps", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%strip_index: index, %acc: f64):
          // The per-element rms_norm reduce core brick: acc_next = acc +
          // (double)(x[i]*x[i]). Its strip_index is the loop induction variable
          // (region arg 0) and its acc is the loop-carried accumulator (region
          // arg 1), the anti-bypass ties. strip_lmul = "m8" anchors the normalize.
          %acc_next = weft_rvv.elementwise_rms_norm_reduce_core %x, %y, %eps, %n strip %strip_index acc %acc {kind = "elementwise_rms_norm_reduce_core", strip_lmul = "m8"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, f64 -> f64
          weft_rvv.typed_elementwise_loop_yield %acc_next : f64
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The rms_norm TU calls scalar libm (1/sqrtf(mean+eps)), so the emitted module
// MUST self-include <math.h> to be a self-contained standalone TU. This pins the
// self-include behavior at the IR level (now keyed on the reduce core brick, not
// the retired monolith op).
// CHECK: emitc.include <"math.h">
// CHECK: emitc.func @weft_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(
// The SCALAR double accumulator (ggml_float sum = 0.0) as an emitc.variable
// lvalue (emitc.for has no iter_args, so the loop-carried accumulator is a
// variable + assign, NOT loop-carried SSA) — the reduce model's f64 acc region
// arg realized as the sum lvalue (load-at-top / assign-at-bottom).
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
// The VECTORIZED normalize strip y[i] = x[i]*scale (strip_lmul m8).
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: for %[[NI:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// The reduction loop, the scalar rsqrt, and the normalize strip are all
// structured emitc nodes (variable/load/mul/cast/add/div/for/call_opaque), not a
// raw C blob. The provenance verbatims carry the constructed reduce brick's op
// identity, NOT the retired monolith op, and NO opaque C blob leaks.
// CHECK-NOT: weft_rvv.ggml_rms_norm_f32
// CHECK-NOT: emitc.verbatim {{.*}}__riscv_v{{.*}};

// The bounded surface is fail-closed on the loop kind, the reduce_map_model fact,
// and the element_sew fact (I7), enforced by the loop-body verifier.
// BADKIND: currently supports only kind "typed_elementwise_loop_body"
// BADMODEL: currently supports only reduce_map_model "map"
// BADSEW: currently supports only element_sew 32
