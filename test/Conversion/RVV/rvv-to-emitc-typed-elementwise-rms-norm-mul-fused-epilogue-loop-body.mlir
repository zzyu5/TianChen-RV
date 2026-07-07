// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "elementwise_mul_map"/kind = "elementwise_bogus_map"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMULKIND
// RUN: sed 's/!tcrv_rvv.vector<f32, "m8">/!tcrv_rvv.vector<f32, "m4">/g' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADCHAINLMUL
// RUN: sed 's/tcrv_rvv.elementwise_mul_map %vy, %w, %z/tcrv_rvv.elementwise_mul_map %vy, %w, %w/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADOUTPUT

// G2 [FUSE] tracer — the CONSTRUCTED fused rms_norm->mul, llama's attn_norm /
// ffn_norm (`ggml_rms_norm` immediately followed by `ggml_mul` against a learned
// weight vector). It BUILDS the L1 chain declaration + L2 region splice on TOP of
// the forward-elementwise reduce scaffold: the rms_norm reduce core brick
// (tcrv_rvv.elementwise_rms_norm_reduce_core) now carries an OPTIONAL single-block
// $epilogue region whose entry argument is the per-strip normalized vector `vy`
// and which holds ONE tcrv_rvv.elementwise_mul_map consumer brick. The reduce-body
// emitter SPLICES the mul into the normalize strip: the normalized vy (a
// register-resident vfmul_vf result) flows STRAIGHT into a per-lane vfmul_vv
// against the loaded weight strip and stores the fused z[] once.
//
// THE FUSION FACT (mechanically checked below): the intermediate normalized row is
// NEVER stored to memory and NEVER reloaded. There is exactly ONE vse32 (the fused
// z[] store) and TWO vle32 (x[] and w[]) in the strip loop; the register-kept vy is
// the FIRST operand of the vfmul_vv, so it is consumed in place with no norm[]
// round-trip (the -2*n*4-byte normalize store + reload of a two-kernel path is
// elided).
//
// BYTE-EXACT (strict, editable-golden argument): the scalar-double Sx^2 fold, the
// mean, and the 1/sqrtf(mean+eps) scale are IDENTICAL to plain rms_norm (verified
// zero-regression against the pre-fusion emit of the unfused reduce-core body). The
// register-kept vy is bit-identical to a store-then-reload of the same f32 vector,
// and `vy * w[i]` is a bare per-lane fp32 multiply at the SAME LMUL anchor (no
// FMA -- no add follows; no cross-lane reduction), so the fused value is byte-exact
// to the non-fused two-op result modulo ONLY the eliminated store/reload.
// Numerical bit-exact-vs-ggml is pending-hardware.

module {
  tcrv.exec.kernel @ggml_rms_norm_mul_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_rms_norm_mul_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "ne00", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %z = tcrv_rvv.runtime_abi_value {c_name = "z", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %eps = tcrv_rvv.runtime_abi_value {c_name = "eps", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_rms_norm_mul_f32, sew = 32 : i64, source_kernel = "ggml_rms_norm_mul_f32_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %z, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%strip_index: index, %acc: f64):
          // The rms_norm reduce core PRODUCER carries the fused mul EPILOGUE: its
          // $epilogue region's entry argument %vy is the per-strip normalized
          // vector (the register-kept vfmul_vf result), and the single
          // tcrv_rvv.elementwise_mul_map consumer multiplies it by the weight
          // strip w[] and stores the fused result to z[] (== the producer output,
          // a single fused destination). anti-bypass: mul strip_index is the loop
          // induction variable (region arg 0); mul chain is %vy (epilogue region
          // arg 0). strip_lmul = "m8" pins the normalize + the mul.
          %acc_next = tcrv_rvv.elementwise_rms_norm_reduce_core %x, %z, %eps, %n strip %strip_index acc %acc {kind = "elementwise_rms_norm_reduce_core", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, f64 -> f64 epilogue {
          ^bb0(%vy: !tcrv_rvv.vector<f32, "m8">):
            tcrv_rvv.elementwise_mul_map %vy, %w, %z, %n strip %strip_index {kind = "elementwise_mul_map"} : !tcrv_rvv.vector<f32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
          }
          tcrv_rvv.typed_elementwise_loop_yield %acc_next : f64
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The fused TU still calls scalar libm (1/sqrtf(mean+eps)), so it self-includes
// <math.h> (keyed on the reduce core brick, unchanged by the epilogue).
// CHECK: emitc.include <"math.h">
// CHECK: emitc.func @tcrv_emitc_ggml_rms_norm_mul_f32_kernel_ggml_rms_norm_mul_f32(
//
// ===== rms_norm reduce is BYTE-IDENTICAL to the unfused path =====
// The scalar-double accumulator + ascending fold (step is the literal 1, NOT a
// vlmax -- unchanged, no vectorized vfredusum).
// CHECK: %[[SUM:.*]] = "emitc.variable"{{.*}}lvalue<!emitc.opaque<"double">>
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[PROD:.*]] = mul %{{.*}}, %{{.*}}{{.*}} -> !emitc.opaque<"float">
// CHECK: cast %[[PROD]] : !emitc.opaque<"float"> to !emitc.opaque<"double">
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// mean = (float)(sum/(double)ne00); scale = 1.0f/sqrtf(mean+eps).
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// CHECK: call_opaque "sqrtf"
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
//
// ===== the FUSED normalize+mul strip (the region splice) =====
// One strip loop (strip_lmul m8): load x, normalize to vy, load w, multiply
// vy*vw, store z -- a SINGLE store, no intermediate norm[] round-trip.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// The x[] strip load + the normalize vfmul_vf -> the register-kept vy.
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: %[[VY:.*]] = call_opaque "__riscv_vfmul_vf_f32m8"
// The mul epilogue provenance verbatim carries the CONSTRUCTED consumer brick's
// identity (elementwise_mul_map), NOT an opaque helper -- the region splice.
// CHECK: route_source_op=tcrv_rvv.elementwise_mul_map
// The w[] strip load, then the FUSED vfmul_vv whose FIRST operand is the
// register-kept vy (proving vy flows straight in -- no norm[] store/reload
// between the normalize and the multiply).
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: %[[VZ:.*]] = call_opaque "__riscv_vfmul_vv_f32m8"(%[[VY]], %{{.*}}, %{{.*}})
// The SINGLE fused store: z[i] = vy * w[i]. The reduce core's normalize store to
// the intermediate row is GONE (the fused kernel writes only z[]).
// CHECK: call_opaque "__riscv_vse32_v_f32m8"(%{{.*}}, %[[VZ]], %{{.*}})
//
// The whole body is structured emitc nodes (variable/load/mul/cast/add/div/for/
// call_opaque), not a raw C blob; the provenance verbatims carry BOTH constructed
// bricks (the reduce core AND the mul epilogue), and NO opaque C blob leaks.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv_v{{.*}};

// The fused mul epilogue brick is fail-closed on its bounded kind, the chain
// LMUL tie (must match the producer normalize strip LMUL), and the single fused
// destination (mul output == producer output), enforced by the verifiers.
// BADMULKIND: currently supports only kind "elementwise_mul_map"
// BADCHAINLMUL: fused epilogue chain block argument must be an f32 RVV vector at the normalize strip LMUL "m8"
// BADOUTPUT: requires the output operand to bind a runtime ABI value of C type 'float *'
