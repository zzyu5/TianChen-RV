module {
  weft.exec.kernel @gelu_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @gelu_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %3 = weft_rvv.setvl %0 {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %3 attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @gelu_f32, sew = 32 : i64, source_kernel = "gelu_f32_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_elementwise_loop_body %1, %2, %0 attributes {element_sew = 32 : i64, kind = "typed_elementwise_loop_body", reduce_map_model = "map"} {
        ^bb0(%arg0: index):
          weft_rvv.elementwise_gelu_map %1, %2, %0 strip %arg0 : index {gelu_precision = "f16lut", kind = "elementwise_gelu_map"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_elementwise_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

