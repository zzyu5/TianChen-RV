// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_elementwise_loop_body"/kind = "plain_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/reduce_map_model = "map"/reduce_map_model = "fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/element_sew = 32 : i64/element_sew = 16 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSEW

// M-FLAT forward-elementwise scaffold (line C, ① 之后) — the CONSTRUCTED f32
// forward-pass scale (y[i] *= v), the FIRST forward-elementwise operator flipped
// dispatch-wired -> constructed (C_construct 28->29). The monolith
// weft_rvv.ggml_vec_scale_f32 op + emitGgmlVecScaleF32 opaque helper were RETIRED;
// the constructed body is the typed elementwise strip-loop op
// (weft_rvv.typed_elementwise_loop_body, reduce_map_model "map") whose region
// carries the per-strip map core brick (weft_rvv.elementwise_scale_map). The loop
// op owns the outer VLEN-robust `for (i = 0; i < n; i += vlmax)` strip loop; the
// brick owns the per-strip `vsetvl_e32m8(n-i) / vle32 / vfmul_vf / vse32`. The
// brick's strip_index MUST be the loop induction variable (region arg 0,
// anti-bypass), so the emit provably addresses y + i, not the loop-invariant
// strip 0.
//
// This is BYTE-EXACT to the retired monolith emit modulo ONLY the source-op
// provenance token (weft_rvv.ggml_vec_scale_f32 -> weft_rvv.elementwise_scale_map):
// a bare per-lane fp32 multiply (no FMA -> -ffp-contract cannot bite; no
// cross-lane reduction -> LMUL/tail/strip-count are correctness-free), so the
// emit is a decomposed pattern-library map primitive with NO opaque hand helper
// ([L-8] constructed-strong). Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_vec_scale_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_scale_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "inout", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %v = weft_rvv.runtime_abi_value {c_name = "v", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_scale_f32, sew = 32 : i64, source_kernel = "ggml_vec_scale_f32_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_elementwise_loop_body %y, %v, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "map", element_sew = 32 : i64, strip_lmul = "m8"} {
        ^bb0(%strip_index: index):
          // The per-strip map core brick: y[i..i+vl] *= v. Its strip_index is the
          // loop induction variable (region arg 0), the anti-bypass tie.
          weft_rvv.elementwise_scale_map %y, %v, %n strip %strip_index : index {kind = "elementwise_scale_map", strip_lmul = "m8"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_elementwise_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(
// The outer setvl config (scope frame) is unchanged: vsetvl_e32m1.
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// The pre-loop VLMAX vsetvl (the map brick's strip anchor, m8).
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// The f32 strip loop over the runtime element count.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Remaining-AVL re-strip vsetvl inside the loop.
// CHECK: sub %arg0, %[[I]]
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// In-place element pointer y + i, cast to float * (anti-bypass: addresses y + i).
// CHECK: add %arg1, %[[I]]
// The f32 load / scalar-broadcast multiply / in-place store chain.
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// The provenance verbatims carry the constructed map brick's op identity, NOT
// the retired monolith op, and NO opaque C blob leaks into the body.
// CHECK-NOT: weft_rvv.ggml_vec_scale_f32
// CHECK-NOT: emitc.verbatim {{.*}}__riscv

// The bounded surface is fail-closed on the loop kind, the reduce_map_model fact,
// and the element_sew fact (I7), enforced by the loop-body verifier.
// BADKIND: currently supports only kind "typed_elementwise_loop_body"
// BADMODEL: currently supports only reduce_map_model "map"
// BADSEW: currently supports only element_sew 32
