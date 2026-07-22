// RUN: weft-opt %s --weft-rvv-materialize-forward-elementwise-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: weft-opt %s --weft-rvv-materialize-forward-elementwise-stream-front-door --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// CERT-FD forward殿后族 (support widening) -- the abstract source op
// weft_rvv.ggml_forward_elementwise (elementwise_model = "cpy") as the
// CONSTRUCT-FROM-ABSTRACT proof of the forward-elementwise FRONT DOOR for the UNARY
// pass-through support op ggml_vec_cpy_f32 (y[i] = x[i]). The pre-emitc pass
// CONSTRUCTS the typed weft_rvv.typed_elementwise_loop_body region
// { elementwise_copy_map; yield } and STOPS before --weft-rvv-lower-to-emitc. The
// emit half re-emits the SHARED byte-exact m8 strip (vsetvl_e32m8 / one vle32 / NO
// combiner / vse32), byte-identical to the dispatch-wired support-op emit
// (rvv-to-emitc-ggml-vec-cpy-f32.mlir) modulo the source-op provenance token
// (weft_rvv.vec_cpy_f32 -> weft_rvv.elementwise_copy_map).

module {
  weft.exec.kernel @vec_cpy_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @vec_cpy_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.ggml_forward_elementwise %x, %y, %n {elementwise_model = "cpy"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// REALIZE: weft_rvv.typed_elementwise_loop_body
// REALIZE: weft_rvv.elementwise_copy_map
// REALIZE: weft_rvv.typed_elementwise_loop_yield
// REALIZE-NOT: weft_rvv.ggml_forward_elementwise

// EMIT: emitc.func @weft_emitc_vec_cpy_f32_kernel_vec_cpy_f32(
// EMIT: route_source_op=weft_rvv.elementwise_copy_map
// EMIT: call_opaque "__riscv_vsetvl_e32m8"
// EMIT: call_opaque "__riscv_vle32_v_f32m8"
// EMIT: call_opaque "__riscv_vse32_v_f32m8"
// EMIT-NOT: __riscv_vfadd_vv_f32m8
// EMIT-NOT: __riscv_vfmul_vv_f32m8
// EMIT-NOT: weft_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
