// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// The bound RVV formula lifecycle constructs the typed elementwise body before
// artifact lowering. The emitter mechanically consumes that body; this test starts
// from the family-local abstract op and checks the complete production path.

module {
  weft.exec.kernel @ggml_vec_scale_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_scale_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "inout", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %v = weft_rvv.runtime_abi_value {c_name = "v", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.ggml_forward_elementwise %y, %v, %n {elementwise_model = "scale"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}


// EMIT: emitc.func @weft_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(
// EMIT: route_source_op=weft_rvv.elementwise_scale_map
// EMIT-NOT: weft_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
