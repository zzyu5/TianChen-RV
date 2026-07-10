// RUN: tcrv-opt %s --tcrv-rvv-materialize-forward-elementwise-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-forward-elementwise-stream-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// CERT-FD forward殿后族 (support widening) -- the abstract source op
// tcrv_rvv.ggml_forward_elementwise (elementwise_model = "cpy") as the
// CONSTRUCT-FROM-ABSTRACT proof of the forward-elementwise FRONT DOOR for the UNARY
// pass-through support op ggml_vec_cpy_f32 (y[i] = x[i]). The pre-emitc pass
// CONSTRUCTS the typed tcrv_rvv.typed_elementwise_loop_body region
// { elementwise_copy_map; yield } and STOPS before --tcrv-rvv-lower-to-emitc. The
// emit half re-emits the SHARED byte-exact m8 strip (vsetvl_e32m8 / one vle32 / NO
// combiner / vse32), byte-identical to the dispatch-wired support-op emit
// (rvv-to-emitc-ggml-vec-cpy-f32.mlir) modulo the source-op provenance token
// (tcrv_rvv.vec_cpy_f32 -> tcrv_rvv.elementwise_copy_map).

module {
  tcrv.exec.kernel @vec_cpy_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @vec_cpy_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @vec_cpy_f32, sew = 32 : i64, source_kernel = "vec_cpy_f32_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.ggml_forward_elementwise %x, %y, %n {elementwise_model = "cpy"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// REALIZE: tcrv_rvv.typed_elementwise_loop_body
// REALIZE: tcrv_rvv.elementwise_copy_map
// REALIZE: tcrv_rvv.typed_elementwise_loop_yield
// REALIZE-NOT: tcrv_rvv.ggml_forward_elementwise

// EMIT: emitc.func @tcrv_emitc_vec_cpy_f32_kernel_vec_cpy_f32(
// EMIT: route_source_op=tcrv_rvv.elementwise_copy_map
// EMIT: call_opaque "__riscv_vsetvl_e32m8"
// EMIT: call_opaque "__riscv_vle32_v_f32m8"
// EMIT: call_opaque "__riscv_vse32_v_f32m8"
// EMIT-NOT: __riscv_vfadd_vv_f32m8
// EMIT-NOT: __riscv_vfmul_vv_f32m8
// EMIT-NOT: tcrv_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
