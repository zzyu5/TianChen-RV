// RUN: weft-opt %s --weft-rvv-materialize-forward-elementwise-stream-front-door | FileCheck %s --check-prefix=REALIZE
// RUN: weft-opt %s --weft-rvv-materialize-forward-elementwise-stream-front-door --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// CERT-FD forward殿后族 -- the abstract source op weft_rvv.ggml_forward_elementwise
// (elementwise_model = "soft_max") as the CONSTRUCT-FROM-ABSTRACT proof of the
// forward-elementwise FRONT DOOR (the forward sibling of the dequant/quant stream
// front doors). The pre-emitc pass CONSTRUCTS the typed
// weft_rvv.typed_elementwise_loop_body region { elementwise_soft_max_reduce_core; yield } and STOPS before
// --weft-rvv-lower-to-emitc, so the certification walker (e5_strong_readout.py) walks
// the REALIZED region. The emit half (emitTypedElementwiseLoopBody) is byte-exact
// UNCHANGED: the emitted C is byte-identical to the hand-authored
// rvv-to-emitc-typed-elementwise-soft-max-reduce-loop-body.mlir emit (proven 0-diff).

module {
  weft.exec.kernel @ggml_vec_soft_max_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_soft_max_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %max = weft_rvv.runtime_abi_value {c_name = "max", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_soft_max_f32, sew = 32 : i64, source_kernel = "ggml_vec_soft_max_f32_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.ggml_forward_elementwise %x, %y, %max, %n {elementwise_model = "soft_max"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// REALIZE: weft_rvv.typed_elementwise_loop_body
// REALIZE: weft_rvv.elementwise_soft_max_reduce_core
// REALIZE: weft_rvv.typed_elementwise_loop_yield
// REALIZE-NOT: weft_rvv.ggml_forward_elementwise

// EMIT: emitc.func @weft_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(
// EMIT: route_source_op=weft_rvv.elementwise_soft_max_reduce_core
// EMIT-NOT: weft_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
