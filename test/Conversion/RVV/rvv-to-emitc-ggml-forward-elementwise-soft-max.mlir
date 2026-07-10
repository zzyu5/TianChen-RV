// RUN: tcrv-opt %s --tcrv-rvv-materialize-forward-elementwise-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-forward-elementwise-stream-front-door --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// CERT-FD forward殿后族 -- the abstract source op tcrv_rvv.ggml_forward_elementwise
// (elementwise_model = "soft_max") as the CONSTRUCT-FROM-ABSTRACT proof of the
// forward-elementwise FRONT DOOR (the forward sibling of the dequant/quant stream
// front doors). The pre-emitc pass CONSTRUCTS the typed
// tcrv_rvv.typed_elementwise_loop_body region { elementwise_soft_max_reduce_core; yield } and STOPS before
// --tcrv-rvv-lower-to-emitc, so the certification walker (e5_strong_readout.py) walks
// the REALIZED region. The emit half (emitTypedElementwiseLoopBody) is byte-exact
// UNCHANGED: the emitted C is byte-identical to the hand-authored
// rvv-to-emitc-typed-elementwise-soft-max-reduce-loop-body.mlir emit (proven 0-diff).

module {
  tcrv.exec.kernel @ggml_vec_soft_max_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_soft_max_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %max = tcrv_rvv.runtime_abi_value {c_name = "max", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_soft_max_f32, sew = 32 : i64, source_kernel = "ggml_vec_soft_max_f32_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.ggml_forward_elementwise %x, %y, %max, %n {elementwise_model = "soft_max"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// REALIZE: tcrv_rvv.typed_elementwise_loop_body
// REALIZE: tcrv_rvv.elementwise_soft_max_reduce_core
// REALIZE: tcrv_rvv.typed_elementwise_loop_yield
// REALIZE-NOT: tcrv_rvv.ggml_forward_elementwise

// EMIT: emitc.func @tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(
// EMIT: route_source_op=tcrv_rvv.elementwise_soft_max_reduce_core
// EMIT-NOT: tcrv_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
