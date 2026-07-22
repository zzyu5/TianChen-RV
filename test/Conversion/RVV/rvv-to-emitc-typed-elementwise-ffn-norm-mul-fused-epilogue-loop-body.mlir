// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "elementwise_mul_map"/kind = "elementwise_bogus_map"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMULKIND
// RUN: sed 's/f32, "m4"/f32, "m8"/g' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADCHAINLMUL

// G2 [FUSE] 铺量 ①a -- REAL llama path #2: ffn_norm. This is the SECOND of the two
// real per-layer rms_norm->mul call-sites in llama's graph (build_llama):
//
//     cur = ggml_rms_norm(ctx, ffn_inp, f_norm_rms_eps);      // ggml_rms_norm
//     cur = ggml_mul(ctx, cur, model.layers[il].ffn_norm);    // ggml_mul by w[]
//
// It reuses the SAME fused rms_norm->mul epilogue mechanism as attn_norm (see
// rvv-to-emitc-typed-elementwise-attn-norm-mul-fused-epilogue-loop-body.mlir): the
// weft_rvv.elementwise_rms_norm_reduce_core producer carries the optional
// single-block $epilogue region whose entry argument is the per-strip normalized
// vector %vy, holding ONE weft_rvv.elementwise_mul_map consumer that multiplies vy
// by the ffn_norm WEIGHT strip and stores the fused result to cur[]. The ONLY
// path-specific facts are the ABI values: the rms_norm input ffn_inp[] (the
// post-attention residual), the learned WEIGHT vector ffn_norm.weight (fed as the
// mul operand), and the fused output cur[]. THE POINT: the learned ffn_norm.weight
// vector is correctly fed into the epilogue multiply (w + i loaded at native
// width, vy * w).
//
// SECOND LMUL ANCHOR (strip_lmul m4, vs attn_norm's m8): the fused epilogue is shown
// here at a DIFFERENT normalize width to pin that the weight vector is consumed at
// the producer's NATIVE strip LMUL regardless of the selected anchor -- the chain
// block-argument vy is an f32 RVV vector at m4, and the mul runs vfmul_vv at m4, so
// vy is consumed at its native width. LMUL is a correctness-free resource knob (no
// cross-lane reduction in the mul), so this stays byte-exact.
//
// THE FUSION FACT (mechanically checked): the intermediate normalized row is NEVER
// stored to memory and NEVER reloaded -- exactly ONE vse32 (the fused cur[] store)
// and TWO vle32 (ffn_inp[] and ffn_norm[]) in the strip loop; the register-kept vy
// is the FIRST operand of the vfmul_vv, so it is consumed in place with no norm[]
// round-trip (the -2*n*4-byte normalize store + reload of a two-kernel path is
// elided).
//
// BYTE-EXACT (strict): the scalar-double Sx^2 fold, the mean, and the
// 1/sqrtf(mean+eps) scale are IDENTICAL to plain rms_norm. The register-kept vy is
// bit-identical to a store-then-reload of the same f32 vector, and
// `vy * ffn_norm[i]` is a bare per-lane fp32 multiply at the SAME LMUL anchor (no
// FMA -- no add follows; no cross-lane reduction). Numerical bit-exact-vs-ggml is
// pending-hardware.

module {
  weft.exec.kernel @ggml_ffn_norm_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_ffn_norm_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "ne00", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %ffn_inp = weft_rvv.runtime_abi_value {c_name = "ffn_inp", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %ffn_norm = weft_rvv.runtime_abi_value {c_name = "ffn_norm", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cur = weft_rvv.runtime_abi_value {c_name = "cur", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %eps = weft_rvv.runtime_abi_value {c_name = "eps", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_elementwise_loop_body %ffn_inp, %cur, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%strip_index: index, %acc: f64):
          // rms_norm(ffn_inp) reduce PRODUCER carrying the fused mul EPILOGUE: the
          // $epilogue entry argument %vy is the per-strip normalized vector (the
          // register-kept vfmul_vf result), and the single elementwise_mul_map
          // multiplies it by the ffn_norm.weight strip and stores to cur[] (== the
          // producer output, the single fused destination). anti-bypass: the mul
          // strip_index is the loop induction variable (region arg 0); the mul
          // chain is %vy (epilogue region arg 0). strip_lmul = "m4".
          %acc_next = weft_rvv.elementwise_rms_norm_reduce_core %ffn_inp, %cur, %eps, %n strip %strip_index acc %acc {kind = "elementwise_rms_norm_reduce_core", strip_lmul = "m4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, f64 -> f64 epilogue {
          ^bb0(%vy: !weft_rvv.vector<f32, "m4">):
            weft_rvv.elementwise_mul_map %vy, %ffn_norm, %cur, %n strip %strip_index {kind = "elementwise_mul_map"} : !weft_rvv.vector<f32, "m4">, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
          }
          weft_rvv.typed_elementwise_loop_yield %acc_next : f64
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The fused ffn_norm TU still calls scalar libm (1/sqrtf(mean+eps)), so it
// self-includes <math.h> (keyed on the reduce core brick, unchanged by the epilogue).
// CHECK: emitc.include <"math.h">
// CHECK: emitc.func @weft_emitc_ggml_ffn_norm_f32_kernel_ggml_ffn_norm_f32(
//
// ===== rms_norm reduce is BYTE-IDENTICAL to plain rms_norm =====
// Scalar-double accumulator + ascending step-1 fold (NO vectorized vfredusum).
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
// ===== the FUSED normalize+mul strip at the m4 anchor (ffn_norm weight fed in) =====
// One strip loop (strip_lmul m4): load ffn_inp, normalize to vy, load ffn_norm,
// multiply vy*ffn_norm, store cur -- a SINGLE store, no intermediate norm[] round-trip.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m4"
// The ffn_inp[] strip load + the normalize vfmul_vf -> the register-kept vy (m4).
// CHECK: call_opaque "__riscv_vle32_v_f32m4"
// CHECK: %[[VY:.*]] = call_opaque "__riscv_vfmul_vf_f32m4"
// The mul epilogue provenance verbatim carries the CONSTRUCTED consumer brick's
// identity (elementwise_mul_map), NOT an opaque helper -- the region splice.
// CHECK: route_source_op=weft_rvv.elementwise_mul_map
// The ffn_norm.weight strip load (w + i) at native m4 width, then the FUSED vfmul_vv
// whose FIRST operand is the register-kept vy (proving vy flows straight in with no
// norm[] store/reload, and the learned ffn_norm.weight vector is the mul operand).
// CHECK: call_opaque "__riscv_vle32_v_f32m4"
// CHECK: %[[VZ:.*]] = call_opaque "__riscv_vfmul_vv_f32m4"(%[[VY]], %{{.*}}, %{{.*}})
// The SINGLE fused store: cur[i] = vy * ffn_norm[i]. The reduce core's normalize
// store to the intermediate row is GONE (the fused kernel writes only cur[]).
// CHECK: call_opaque "__riscv_vse32_v_f32m4"(%{{.*}}, %[[VZ]], %{{.*}})
//
// Structured emitc nodes only; the provenance verbatims carry BOTH constructed
// bricks (the reduce core AND the mul epilogue); NO opaque C blob leaks.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv_v{{.*}};

// The fused mul epilogue brick is fail-closed on its bounded kind and on the chain
// LMUL tie (the chain block arg must match the producer normalize strip LMUL m4, so
// vy is consumed at its native width), enforced by the verifiers.
// BADMULKIND: currently supports only kind "elementwise_mul_map"
// BADCHAINLMUL: fused epilogue chain block argument must be an f32 RVV vector at the normalize strip LMUL "m4"
